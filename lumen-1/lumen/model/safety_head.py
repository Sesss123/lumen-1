"""Safety classifier head on decoder hidden states."""

import torch
import torch.nn as nn

from lumen.model.config import LumenConfig

HARM_CATEGORIES = [
    "violence",
    "sexual",
    "hate",
    "self_harm",
    "illegal",
    "harassment",
    "misinformation",
    "privacy",
]


class SafetyHead(nn.Module):
    """Auxiliary 2-layer MLP classifier for 8 harm categories."""

    def __init__(self, config: LumenConfig):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, config.num_safety_categories),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, seq_len, hidden_size) or (B, hidden_size)
        Returns:
            logits: (B, num_categories) harm scores
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        return self.classifier(hidden_states)

    @property
    def category_names(self) -> list[str]:
        return HARM_CATEGORIES
