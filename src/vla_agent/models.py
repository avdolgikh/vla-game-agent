"""Model architectures for the Crafter MVP-1 vision-only agent."""

from __future__ import annotations

import torch
from torch import nn


class CrafterCNN(nn.Module):
    """Minimal CNN encoder for 64x64 RGB observations -> action logits."""

    def __init__(self, num_actions: int = 8) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_actions)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
