# import the necessary libraries
from collections import defaultdict
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformers
from matplotlib import rc
from pylab import rcParams
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertModel, BertTokenizer, get_constant_schedule_with_warmup

sns.set(style="whitegrid", palette="muted", font_scale=1.2)

HAPPY_COLORS_PALETTE = [
    "#01BEFE",
    "#FFDD00",
    "#FF7D00",
    "#FF006D",
    "#ADFF02",
    "#8F00FF",
]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams["figure.figsize"] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


PRE_TRAINED_MODEL_NAME = "bert-base-cased"


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_len=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class GooglePlayReviewDataset(GPReviewDataset):
    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size

    df_train, df_test = train_test_split(
        self.df, test_size=0.1, random_state=RANDOM_SEED
    )
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = GPReviewDataset(
            reviews=df.content.to_numpy(),
            targets=df.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return DataLoader(ds, batch_size=self.batch_size, num_workers=4)

    def get_splitted_data(self):
        train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.max_len, self.batch_size
        )
        val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.max_len, self.batch_size
        )
        test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.max_len, self.batch_size
        )

        return train_data_loader, val_data_loader, test_data_loader


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


df = pd.read_csv("../notebooks/reviews.csv")
gprd = GooglePlayReviewDataset(df, 16)
train_data_loader, val_data_loader, test_data_loader = gprd.get_splitted_data()
model = SentimentClassifier(3)
model = model.to_device(device)
EPOCHS = (10,)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_constant_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
loss_function = nn.CrossEntropyLoss().to(device)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


if __name__ == "__main__":
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_function, optimizer, device, scheduler,
        )
        print(f"Train loss {train_loss} accuracy {train_acc}")
        val_acc, val_loss = eval_model(model, val_data_loader, loss_function, device)
        print(f"Val   loss {val_loss} accuracy {val_acc}")
        print()
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "best_model_state.bin")
            best_accuracy = val_acc
