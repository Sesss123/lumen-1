"""Lumen-1 decoder transformer blocks."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from lumen.model.attention import HybridLocalGlobalAttention
from lumen.model.config import LumenConfig
from lumen.model.norms import RMSNorm, SwiGLU


class DecoderLayer(nn.Module):
    def __init__(self, config: LumenConfig, layer_idx: int):
        super().__init__()
        self.self_attn = HybridLocalGlobalAttention(config, layer_idx)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, config.dropout)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present


class LumenDecoder(nn.Module):
    """32-layer decoder-only transformer backbone."""

    def __init__(self, config: LumenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        bsz, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        presents = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None

            if self.config.use_gradient_checkpointing and self.training:
                hidden_states, present = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past,
                    use_cache=use_cache,
                )

            if use_cache:
                presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents
