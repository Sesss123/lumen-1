"""FSDP training loop for Lumen-1."""

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from lumen.model.config import LumenConfig, ModelSize
from lumen.model.lumen_model import LumenForCausalLM
from lumen.train.schedulers import get_cosine_schedule, get_wsd_schedule


def load_config(path: str) -> LumenConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    valid = {f.name for f in fields(LumenConfig)}
    filtered = {k: v for k, v in raw.items() if k in valid}
    if "model_size" in filtered:
        filtered["model_size"] = ModelSize(filtered["model_size"])
    return LumenConfig(**filtered)


class LumenTrainer:
    """Trainer with optional FSDP wrapping and multimodal loss."""

    def __init__(
        self,
        model: LumenForCausalLM,
        config: LumenConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_steps: int = 100_000,
        warmup_steps: int = 2000,
        stable_steps: int = 80_000,
        decay_steps: int = 18_000,
        schedule: str = "wsd",
        output_dir: str = "checkpoints",
        use_fsdp: bool = False,
        gradient_accumulation_steps: int = 1,
        log_every: int = 100,
        save_every: int = 1000,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_every = log_every
        self.save_every = save_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if use_fsdp and torch.cuda.is_available():
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from lumen.model.decoder import DecoderLayer

            auto_wrap = transformer_auto_wrap_policy({DecoderLayer})
            self.model = FSDP(model, auto_wrap_policy=auto_wrap)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )

        if schedule == "wsd":
            self.scheduler = get_wsd_schedule(
                self.optimizer, warmup_steps, stable_steps, decay_steps, peak_lr=learning_rate
            )
        else:
            self.scheduler = get_cosine_schedule(self.optimizer, warmup_steps, max_steps)

        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    def _compute_multimodal_loss(
        self,
        outputs,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = outputs.loss
        if loss is None:
            return torch.tensor(0.0, device=self.device)
        return loss

    def train_step(self, batch: Dict[str, Any]) -> float:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = self._compute_multimodal_loss(outputs) / self.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps

    def train(self) -> None:
        data_iter = iter(self.train_dataloader)
        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            loss = self.train_step(batch)

            if self.global_step % self.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"step={self.global_step} loss={loss:.4f} lr={lr:.2e}")

            if self.global_step % self.save_every == 0 and self.global_step > 0:
                self.save_checkpoint(f"step_{self.global_step}")

            self.global_step += 1

        self.save_checkpoint("final")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)
            outputs = self.model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            n += 1
        return {"val_loss": total_loss / max(n, 1), "perplexity": torch.exp(torch.tensor(total_loss / max(n, 1))).item()}

    def save_checkpoint(self, name: str) -> None:
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        state = self.model.state_dict() if not hasattr(self.model, "module") else self.model.module.state_dict()
        torch.save(state, path / "model.pt")
        torch.save({"step": self.global_step, "optimizer": self.optimizer.state_dict()}, path / "trainer.pt")
        print(f"Checkpoint saved: {path}")
