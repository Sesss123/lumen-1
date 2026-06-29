"""Modality projectors mapping encoder outputs to decoder hidden size."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjector(nn.Module):
    """Two-layer GELU MLP projector (vision/audio -> decoder dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x)
