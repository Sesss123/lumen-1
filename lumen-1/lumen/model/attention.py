"""Hybrid Local-Global (HLG) attention with Grouped Query Attention."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.model.config import LumenConfig
from lumen.model.norms import mup_init_linear
from lumen.model.rope import RotaryEmbedding, apply_rotary_pos_emb

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _build_swa_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Causal sliding-window mask: attend within window_size tokens."""
    idx = torch.arange(seq_len, device=device)
    dist = idx.unsqueeze(1) - idx.unsqueeze(0)
    mask = (dist >= 0) & (dist < window_size)
    attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    attn_mask.masked_fill_(~mask, float("-inf"))
    return attn_mask


def _build_gsa_mask(
    seq_len: int,
    num_sink_tokens: int,
    stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Global Sink Attention: causal + sink tokens + sparse global stride positions."""
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    global_cols = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    positions = torch.arange(0, seq_len, stride, device=device)
    global_cols[:, positions] = True
    # First num_sink_tokens are always attendable (sink pattern)
    if num_sink_tokens > 0:
        sink = min(num_sink_tokens, seq_len)
        global_cols[:, :sink] = True
    mask = causal | global_cols
    attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    attn_mask.masked_fill_(~mask, float("-inf"))
    return attn_mask


class HybridLocalGlobalAttention(nn.Module):
    """GQA attention with sliding-window or global-sink modes per layer."""

    def __init__(self, config: LumenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_key_value_groups
        self.is_global = config.is_global_attention_layer(layer_idx)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
        )

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            mup_init_linear(proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(query, position_ids=position_ids, seq_len=q_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present_key_value = (key, value) if use_cache else None

        if (
            self.config.use_flash_attention
            and HAS_FLASH_ATTN
            and not self.is_global
            and attention_mask is None
            and hidden_states.device.type == "cuda"
        ):
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=(self.config.sliding_window_size, 0),
            )
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        else:
            key = _repeat_kv(key, self.num_kv_groups)
            value = _repeat_kv(value, self.num_kv_groups)
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim ** 0.5)

            if attention_mask is None:
                if self.is_global:
                    local_mask = _build_gsa_mask(
                        key.size(2),
                        self.config.global_sink_tokens,
                        self.config.global_attention_stride,
                        hidden_states.device,
                        attn_weights.dtype,
                    )
                else:
                    local_mask = _build_swa_mask(
                        key.size(2),
                        self.config.sliding_window_size,
                        hidden_states.device,
                        attn_weights.dtype,
                    )
                attn_weights = attn_weights + local_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = torch.matmul(attn_weights, value)
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                bsz, q_len, self.num_heads * self.head_dim
            )

        return self.o_proj(attn_output), present_key_value
