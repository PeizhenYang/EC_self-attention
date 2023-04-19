import torch
import torch.nn as nn


class TemporalWeightedSum(nn.Module):
    def __init__(self, n_channels):
        super(TemporalWeightedSum, self).__init__()
        self.fc1 = nn.Linear(n_channels, n_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_channels, 1)

    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(x)))           # [batch_size, seq_len, 1]
        a = torch.softmax(a, dim=1)
        h_hat = torch.sum(a*x, dim=1) # [batch_size, n_channels]
        return h_hat