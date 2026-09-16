"""Reusable fully-connected Physics-Informed Neural Network."""

import torch
from torch import nn


class PINN(nn.Module):
    """MLP mapping space-time coordinates (x, t) to a scalar field T."""

    def __init__(self, input_dim=2, output_dim=1, hidden_dim=64, hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize()

    def _initialize(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)
