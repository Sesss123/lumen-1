"""Dataset blending and weighted sampling."""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class DatasetConfig:
    name: str
    path: str
    weight: float = 1.0
    modality: str = "text"  # text, image_text, audio


class DatasetBlender(IterableDataset):
    """Weighted mixture of dataset shards for pretraining."""

    def __init__(self, configs: List[DatasetConfig], seed: int = 42):
        self.configs = configs
        self.seed = seed
        total = sum(c.weight for c in configs)
        self.normalized_weights = [c.weight / total for c in configs]

    def __iter__(self) -> Iterator[dict]:
        import random

        rng = random.Random(self.seed)
        while True:
            config = rng.choices(self.configs, weights=self.normalized_weights, k=1)[0]
            yield {"source": config.name, "modality": config.modality, "path": config.path}


def blend_datasets(
    configs: List[DatasetConfig],
    batch_size: int = 32,
) -> DatasetBlender:
    return DatasetBlender(configs)
