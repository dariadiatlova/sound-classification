import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import wandb

from typing import Optional, List

from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from omegaconf import OmegaConf

from src.dataset import get_dataloader, SoundDS
from src.wandb_init import _init_wandb


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
            Abstract class for LSTM cells.
            :param input_dim: int input channels dimension.
            :param hidden_dim: int dimensionality for hidden layers.
            :param output_dim: int number of classes.
            :param num_layers: int number of lstm layers.
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.tensor, hidden: Optional[torch.tensor] = None) -> torch.tensor:
        lstm_out, hidden = self.lstm(x, hidden)
        logits = self.linear(lstm_out[:, -1, :])
        predicted_class_probabilities = F.log_softmax(logits, dim=1)
        return predicted_class_probabilities, hidden


# lstm = LSTM(344, 256, 10, 1)
# dataset = SoundDS(1, test=False)
# pred, hid = lstm(torch.unsqueeze(dataset[0][0], 0))
# print(pred)


class Classifier(pl.LightningModule):
    def __init__(self, accuracies: [List[Optional[int]]], val_losses: [List[Optional[int]]],
                 train_loss: List[Optional[int]], train_config: dict):
        super(Classifier, self).__init__()
        self.accuracies = accuracies
        self.train_loss = train_loss
        self.val_losses = val_losses

        self.train_config = train_config

        self.lr = self.train_config["lr"]
        self.model = LSTM(self.train_config["input_dim"], self.train_config["hidden_dim"],
                          self.train_config["output_dim"], self.train_config["num_layers"])
        self.loss_function = nn.NLLLoss()
        self.stateful = False
        self.hidden = None

    def forward(self, x, hidden=None):
        prediction, self.hidden = self.model(x, hidden)
        return prediction

    def training_step(self, batch, batch_idx):
        x, labels = batch
        y_pred, self.hidden = self.model(x, self.hidden)
        if not self.stateful:
            self.hidden = None
        else:
            h_0, c_0 = self.hidden
            h_0.detach_(), c_0.detach_()
            self.hidden = (h_0, c_0)
        loss = self.loss_function(y_pred, labels)
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        y_pred, self.hidden = self.model(x, self.hidden)
        if not self.stateful:
            self.hidden = None

        _, val_loss, acc = self._get_predictions_loss_accuracy(batch)

        self.accuracies.append(acc)
        self.val_losses.append(val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def _get_predictions_loss_accuracy(self, batch):
        x, y = batch
        logits = self(x)
        predictions = torch.argmax(logits, dim=1)
        loss = self.loss_function(logits, y)
        acc = accuracy(predictions, y)
        return predictions, loss, acc


def main(config_path: str = "config.yaml"):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    train_config = config["train"]
    wandb_config = config["wandb"]

    _init_wandb(wandb_config, train_config)

    for epoch in range(train_config["epochs"]):
        accuracies, val_losses, train_losses = [], [], []

        for test_folder in range(1, 10):
            train_loader, test_loader = get_dataloader(test_folder)
            model = Classifier(accuracies, val_losses, train_losses, train_config)
            checkpoint_callback = ModelCheckpoint(filepath=f"checkpoints/{epoch}/test_fold_{test_folder}")
            trainer = pl.Trainer(max_epochs=1, logger=False, checkpoint_callback=checkpoint_callback)
            trainer.fit(model, train_loader, test_loader)

            accuracies = model.accuracies
            val_losses = model.val_losses
            train_losses = model.train_losses

            # as recommended in common pitfalls section, evaluate model using k-fold evaluation
            if test_folder == 9:
                wandb.log({"train_loss": np.mean(train_losses),
                           "val_loss": np.mean(val_losses),
                           "val_acc": np.mean(accuracies)})


if __name__ == "__main__":
    main()
