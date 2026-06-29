"""Knowledge distillation: 7B teacher -> 3B/1B students."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.model.config import LumenConfig, ModelSize
from lumen.model.lumen_model import LumenForCausalLM

LAYER_MAP_7B_TO_3B = {i: int(i * 24 / 32) for i in range(32)}
LAYER_MAP_7B_TO_1B = {i: int(i * 16 / 32) for i in range(32)}


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(-1)
    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum().clamp(min=1)
    return kl.mean()


def hidden_state_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    proj: Optional[nn.Linear] = None,
) -> torch.Tensor:
    if proj is not None:
        teacher_hidden = proj(teacher_hidden)
    return F.mse_loss(student_hidden, teacher_hidden)


def build_student(size: ModelSize) -> LumenForCausalLM:
    return LumenForCausalLM.from_size(size)


class Distiller:
    """Distills Lumen-1-7B teacher into 3B or 1B student."""

    def __init__(
        self,
        teacher: LumenForCausalLM,
        student: LumenForCausalLM,
        layer_map: Dict[int, int],
        temperature: float = 2.0,
        alpha_kl: float = 0.5,
        alpha_hidden: float = 0.3,
        alpha_ce: float = 0.2,
        device: Optional[str] = None,
    ):
        self.teacher = teacher
        self.student = student
        self.layer_map = layer_map
        self.temperature = temperature
        self.alpha_kl = alpha_kl
        self.alpha_hidden = alpha_hidden
        self.alpha_ce = alpha_ce
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.teacher.to(self.device).eval()
        self.student.to(self.device).train()
        for p in self.teacher.parameters():
            p.requires_grad = False

        t_dim = teacher.config.hidden_size
        s_dim = student.config.hidden_size
        self.hidden_proj = (
            nn.Linear(t_dim, s_dim, bias=False).to(self.device) if t_dim != s_dim else None
        )

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor):
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            teacher_out = self.teacher(input_ids=input_ids, return_safety_logits=False)

        student_out = self.student(input_ids=input_ids, labels=labels, return_safety_logits=False)

        kl = kl_divergence_loss(
            student_out.logits,
            teacher_out.logits,
            temperature=self.temperature,
        )

        ce = student_out.loss or F.cross_entropy(
            student_out.logits[:, :-1, :].reshape(-1, student_out.logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        hidden_loss = torch.tensor(0.0, device=self.device)
        if self.alpha_hidden > 0 and student_out.hidden_states is not None:
            hidden_loss = hidden_state_loss(
                student_out.hidden_states,
                teacher_out.hidden_states,
                self.hidden_proj,
            )

        loss = self.alpha_kl * kl + self.alpha_ce * ce + self.alpha_hidden * hidden_loss
        return loss, loss.item()

    def train(self, dataloader, optimizer, max_steps: int = 50_000, log_every: int = 100) -> None:
        step = 0
        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = input_ids.clone()
            loss, loss_val = self.train_step(input_ids, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                print(f"distill step={step} loss={loss_val:.4f}")
            step += 1
            if step >= max_steps:
                break
