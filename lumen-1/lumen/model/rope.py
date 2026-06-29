"""Rotary position embeddings with YaRN extrapolation support."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_linear_ramp_mask(
    min_val: float, max_val: float, dim: int, device: torch.device
) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (
        max_val - min_val
    )
    return torch.clamp(linear_func, 0, 1)


class RotaryEmbedding(nn.Module):
    """RoPE with optional YaRN scaling for context extrapolation."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if scaling_factor > 1.0:
            self._init_yarn(scaling_factor)

    def _init_yarn(self, factor: float) -> None:
        dim = self.dim
        beta_fast = 32
        beta_slow = 1
        orig_max = self.max_position_embeddings
        extrapolated = orig_max * factor

        pos_freqs = self.base ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=self.inv_freq.device) / dim
        )
        inv_freq_extrap = 1.0 / pos_freqs
        inv_freq_inter = 1.0 / (factor * pos_freqs)

        low = _yarn_find_correction_dim(beta_fast, dim, self.base, orig_max)
        high = _yarn_find_correction_dim(beta_slow, dim, self.base, orig_max)
        mask = _yarn_linear_ramp_mask(low, high, dim // 2, inv_freq_extrap.device).unsqueeze(0)

        inv_freq = inv_freq_inter * (1 - mask) + inv_freq_extrap * mask
        self.register_buffer("inv_freq", inv_freq.squeeze(0), persistent=False)
        self.register_buffer(
            "yarn_attn_factor",
            torch.tensor(math.log(factor) / math.log(extrapolated / orig_max) + 1.0, device=self.inv_freq.device),
            persistent=False,
        )
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2] if position_ids is None else position_ids.max().item() + 1

        if position_ids is not None:
            t = position_ids.float().reshape(-1)
            if t.numel() == seq_len:
                t = t
            else:
                t = position_ids.float()
        else:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)

        if t.dim() > 1:
            # (batch, seq) position ids — compute per-position freqs
            freqs = torch.einsum("bs,d->bsd", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(1)
            sin = emb.sin().unsqueeze(1)
        else:
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(0)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)

        if hasattr(self, "yarn_attn_factor"):
            cos = cos * self.yarn_attn_factor
            sin = sin * self.yarn_attn_factor

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 4 and cos.shape[0] == 1:
        # Standard (1, 1, seq, dim) broadcast
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        # (batch, 1, seq, dim) from batched position ids
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
