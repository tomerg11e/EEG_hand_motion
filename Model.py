import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import log_softmax, softmax


class Model(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int, num_layers: int, seq_length: int, num_classes: int = 3):
        super().__init__()
        self.in_feature = in_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=in_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def create_zero_hidden(self, x):
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size))
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size))
        return h_0, c_0

    def forward(self, x, hidden):
        if hidden is None:
            hidden = self.create_zero_hidden(x)

        # out, (h, c) = self.lstm(x, hidden)
        out, (h, c) = self.lstm(x)
        out = self.head(torch.tanh(h[-1]))
        out = log_softmax(out, dim=1)
        return out, h
