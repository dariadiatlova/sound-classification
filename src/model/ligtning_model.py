from typing import List, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import accuracy

from src.model.lstm import LSTM


class Classifier(pl.LightningModule):
    def __init__(self, train_config: dict, inference: bool = False):
        super(Classifier, self).__init__()
        self.train_config = train_config

        self.lr = self.train_config["lr"]
        self.model = LSTM(self.train_config["input_dim"], self.train_config["hidden_dim"],
                          self.train_config["output_dim"], self.train_config["num_layers"],
                          self.train_config["dropout"], inference=inference)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        prediction = self.model(x)
        return prediction

    def training_step(self, batch, batch_idx):
        x, labels = batch
        self.model.train()
        y_pred = self.model(x)
        loss = self.loss_function(y_pred, labels)
        train_accuracy = self._compute_accuracy(labels, y_pred)
        log_dictionary = {"train/loss": loss, "train/accuracy": train_accuracy}
        self.log_dict(log_dictionary, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        self.model.eval()
        y_pred = self.model(x)
        loss = self.loss_function(y_pred, labels)
        val_accuracy = self._compute_accuracy(labels, y_pred)
        return {"loss": loss, "accuracy": val_accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        loss = torch.stack([i["loss"] for i in outputs]).mean()
        acc = torch.stack([i["accuracy"] for i in outputs]).mean()
        log_dictionary = {"val/loss": loss, "val/accuracy": acc}
        self.log_dict(log_dictionary, on_step=False, on_epoch=True)

    @staticmethod
    def _compute_accuracy(y_true, logits):
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            acc = accuracy(predictions, y_true)
            return acc
