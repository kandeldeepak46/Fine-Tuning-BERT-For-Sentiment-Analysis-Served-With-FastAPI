import json

from torch import nn
from transformers import BertModel

with open("config.json") as json_file:
    config = json.load(json_file)


class SentimentClassifier(nn.Module):
    """A blueprint about the structure of the model we are using for the sentiment classifier, initialized with BertModel. 
       Dropout used for regularization.
       Linear activation function used in classifing layer.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of classes
        """
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config["BERT_MODEL"])
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        """
        :param input_ids: input_ids of the sentence
        :param attention_mask: attention_mask of the sentence
        :return: output of the classifier
        """
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)
