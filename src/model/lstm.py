from typing import Tuple

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        """
            Abstract class for LSTM cells.
            :param input_dim: expected number of features, mel freq_bands, 64.
            :param hidden_dim: int dimensionality for hidden layers, 256.
            :param output_dim: int number of classes, 10.
            :param num_layers: int number of lstm layers.
        """
        super(LSTM, self).__init__()
        # input: [batch_size, mel_freq_bands, time_steps], [16, 64, 344]
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        # non-stateful hidden as we do not implement any inference here
        lstm_out, _ = self.lstm(x)
        logits = self.linear(lstm_out[:, -1, :])
        return logits
