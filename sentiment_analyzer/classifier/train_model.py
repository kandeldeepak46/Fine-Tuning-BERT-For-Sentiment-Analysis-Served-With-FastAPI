import os
from collections import defaultdict
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformers
from matplotlib import rc
from loguru import logger
from pylab import rcParams
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    BertModel,
    BertTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


PRE_TRAINED_MODEL_NAME = "bert-base-cased"
RANDOM_SEED = 42
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 10
TRAINING_FILE_PATH = "./notebooks/reviews.csv"
HAPPY_COLORS_PALETTE = [
    "#01BEFE",
    "#FFDD00",
    "#FF7D00",
    "#FF006D",
    "#ADFF02",
    "#8F00FF",
]


sns.set(style="whitegrid", palette="muted", font_scale=1.2)
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams["figure.figsize"] = 12, 8
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ["negative", "neutral", "positive"]


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item) -> dict:
        """
        :param item: index of the review
        :return: dictionary with the review and the target
        """
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


def to_sentiment(rating) -> int:
    """
    :param rating: rating of the review
    :return: sentiment of the review
    """
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


if not os.path.exists(TRAINING_FILE_PATH):
    raise FileExistsError("Training file not found, please check the directory")

df = pd.read_csv(TRAINING_FILE_PATH)
df["sentiment"] = df.score.apply(to_sentiment)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


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
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)


def create_data_loader(df, tokenizer, max_len, batch_size):
    """
    create data loader for training and validation data sets and test data set for testing the model performance on test data set 
    """
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):

        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: input ids of the review
        :param attention_mask: attention mask of the review
        :return: logits of the review
        """
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)


model = SentimentClassifier(len(class_names))
model = model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(
    model, data_loader, loss_fn, optimizer, device, scheduler, n_examples
) -> list:
    """
    :param model: model to be trained
    :param data_loader: data loader for training data set
    :param loss_fn: loss function
    :param optimizer: optimizer for the model
    :param device: device to train the model
    :param scheduler: scheduler for the model
    :param n_examples: number of examples in the training data set
    :return: list of training loss and validation loss
    """
    model = model.train()
    losses = []
    correct_predictions = 0
    try:

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

    except Exception as e:
        raise ValueError("could not train the module")


def eval_model(model, data_loader, loss_fn, device, n_examples) -> list:
    """
    :param model: model to be evaluated
    :param data_loader: data loader for validation data set
    :param loss_fn: loss function
    :param device: device to evaluate the model
    :param n_examples: number of examples in the validation data set
    :return: list of validation loss and validation accuracy
    """
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


def get_predictions(model, data_loader):
    """
    :param model: model to be evaluated
    :param data_loader: data loader for validation data set
    :return: list of predictions
    """
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix):
    """
    :param confusion_matrix: confusion matrix
    :return: confusion matrix plot
    """
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True sentiment")
    plt.xlabel("Predicted sentiment")


def run_experiment(
    plot_accuracy=False,
    plot_loss=False,
    model_evaluation=False,
    show_confusion_matrix=False,
):
    """
    Function to train the sentiment classifier model for 3 class of sentiment

    Arguments:
            plot_arruracy <bool> : if True, shows the accuracy graph after completion of training
            plos_loss<bool> : if True, shows the loss graph after completion of training
            model_evaluaion<bool> : if True, performance of the model is analysed in validation dataset
            show_confusion_matrix<bool> : if True, the confusion matrix graph is plotted
    Returns:
            None
    """
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train),
        )
        logger.info(f"Train loss {train_loss} accuracy {train_acc}")
        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn, device, len(df_val)
        )

        logger.info(f"Val   loss {val_loss} accuracy {val_acc}")
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "best_model_state.bin")
            best_accuracy = val_acc

    if plot_accuracy is True:
        plt.plot(history["train_acc"], label="train accuracy")
        plt.plot(history["val_acc"], label="validation accuracy")
        plt.title("Training history")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.ylim([0, 1])

    if plot_loss is True:
        plt.plot(history["train_loss"], label="train loss")
        plt.plot(history["val_loss"], label="validation loss")
        plt.title("Loss history")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.ylim([0, 1])

    if model_evaluation is True:

        test_acc, _ = eval_model(model, test_data_loader, loss_fn, device, len(df_test))

        y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
            model, test_data_loader
        )
        print(classification_report(y_test, y_pred, target_names=class_names))

    if show_confusion_matrix is True:
        y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
            model, test_data_loader
        )
        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        show_confusion_matrix(df_cm)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
