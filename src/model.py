import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import wandb

from typing import Optional, List
from datetime import datetime

from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from omegaconf import OmegaConf

from src.dataset import get_dataloader, SoundDS


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
                 config_path: str = "config.yaml", log_acc: bool = False):
        super(Classifier, self).__init__()
        self.accuracies = accuracies
        self.val_losses = val_losses
        self.log_acc = log_acc

        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)

        self.train_config = config["train"]
        self.wandb_config = config["wandb"]
        self.save_hyperparameters()
        self._init_wandb()

        self.lr = self.train_config["lr"]
        self.model = LSTM(config["train"]["input_dim"], config["train"]["hidden_dim"], config["train"]["output_dim"],
                          config["train"]["num_layers"])
        self.loss_function = nn.NLLLoss()
        self.stateful = False
        self.hidden = None

    def _init_wandb(self):
        if self.wandb_config["use_wandb"]:
            wandb.login()
            now = datetime.now()
            date_time = f"{now.hour}:{now.minute}:{now.second}-{now.day}.{now.month}.{now.year}"

            wandb.init(project=self.wandb_config["project"],
                       name=f'{self.wandb_config["name"]}:{date_time}',
                       notes=self.wandb_config["notes"],
                       entity=self.wandb_config["entity"],
                       config=self.train_config)

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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        y_pred, self.hidden = self.model(x, self.hidden)
        if not self.stateful:
            self.hidden = None

        _, val_loss, acc = self._get_predictions_loss_accuracy(batch)

        self.accuracies.append(acc)
        self.val_losses.append(val_loss)

        if self.log_acc:
            self.log("val_acc", np.mean(self.accuracies), prog_bar=True)
            self.log("val_loss", np.mean(self.val_losses), prog_bar=True)

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


def main():
    for epoch in range(5):
        accuracies = []
        val_losses = []
        log_acc = False
        for test_folder in range(1, 10):
            train_loader, test_loader = get_dataloader(test_folder)
            # print(epoch, test_folder, len(train_loader), len(test_loader))
            if test_folder == 9:
                log_acc = True
            model = Classifier(accuracies, val_losses, log_acc=log_acc)
            wandb_logger = WandbLogger(project="sound-classification")
            checkpoint_callback = ModelCheckpoint(filepath=f"checkpoints/{epoch}/test_fold_{test_folder}")
            trainer = pl.Trainer(max_epochs=1, logger=wandb_logger, checkpoint_callback=checkpoint_callback)
            trainer.fit(model, train_loader, test_loader)
            accuracies = model.accuracies
            val_losses = model.val_losses


if __name__ == "__main__":
    main()
