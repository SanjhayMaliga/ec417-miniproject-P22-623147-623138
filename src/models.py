"""Persistence, a ridge linear baseline, and matched GRU variants."""
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge

def persistence(dataset):
    return np.repeat(dataset.temperature[dataset.origins, None], len(dataset.horizons), axis=1)

def linear_model(alpha):
    return Ridge(alpha=alpha, solver="cholesky")

class Forecaster(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])
