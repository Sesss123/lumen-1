"""PyTorch datasets for text and multimodal training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from lumen.data.filters import DedupFilter, PIIFilter, QualityFilter, SafetyFilter


@dataclass
class MultimodalSample:
    text: str
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LumenDataset(Dataset):
    """Loads preprocessed Parquet/JSONL shards with filtering pipeline."""

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 4096,
        apply_filters: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[dict] = []

        dedup = DedupFilter() if apply_filters else None
        quality = QualityFilter() if apply_filters else None
        safety = SafetyFilter() if apply_filters else None
        pii = PIIFilter() if apply_filters else None

        if self.data_path.is_file():
            self._load_file(self.data_path, dedup, quality, safety, pii)
        elif self.data_path.is_dir():
            for f in sorted(self.data_path.glob("*.jsonl")):
                self._load_file(f, dedup, quality, safety, pii)

    def _load_file(self, path, dedup, quality, safety, pii) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record.get("text", "")
                if dedup and dedup.is_duplicate(text):
                    continue
                if quality and not quality.passes(text):
                    continue
                if safety and not safety.passes(text):
                    continue
                if pii:
                    text = pii.redact(text)
                    record["text"] = text
                self.samples.append(record)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.samples[idx]
        result = {"text": record.get("text", "")}

        if self.tokenizer:
            ids = self.tokenizer.encode(record["text"], add_eos=True)
            ids = ids[: self.max_length]
            result["input_ids"] = torch.tensor(ids, dtype=torch.long)
            result["labels"] = result["input_ids"].clone()

        if "image_path" in record:
            result["image_path"] = record["image_path"]
        if "audio_path" in record:
            result["audio_path"] = record["audio_path"]

        return result
