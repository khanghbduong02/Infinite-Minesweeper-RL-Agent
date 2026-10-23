"""Spatial feature extractor for the Dict observations used by MaskablePPO."""
from __future__ import annotations

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn


class MinesweeperCNNExtractor(BaseFeaturesExtractor):
    """Shared spatial CNN for the board image; masks are applied to actions."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        channels, height, width = observation_space["obs"].shape
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened = self.cnn(torch.zeros(1, channels, height, width)).shape[1]
        self.linear = nn.Sequential(nn.Linear(flattened, features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.linear(self.cnn(observations["obs"]))
