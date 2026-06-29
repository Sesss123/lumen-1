"""Lumen-1 model configuration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelSize(str, Enum):
    DRAFT_1B = "1b"
    EDGE_3B = "3b"
    PRIMARY_7B = "7b"


@dataclass
class LumenConfig:
    """Hyperparameters for Lumen-1 decoder and multimodal stack."""

    # Decoder
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 14336
    vocab_size: int = 128256
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rope_scaling_factor: float = 4.0  # YaRN extrapolation to 128K
  # HLG attention
    sliding_window_size: int = 4096
    global_sink_tokens: int = 4
    global_attention_stride: int = 4096
    global_attention_layers: tuple = tuple(i for i in range(4, 32, 8))

    # Vision encoder (ViT-B/16)
    vision_hidden_size: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vision_patch_size: int = 16
    vision_image_size: int = 384
    vision_intermediate_size: int = 3072

    # Audio encoder (Conformer-S)
    audio_hidden_size: int = 512
    audio_num_layers: int = 12
    audio_num_heads: int = 8
    audio_conv_kernel_size: int = 31
    audio_num_mel_bins: int = 80
    audio_sample_rate: int = 16000
    audio_tokens_per_second: int = 50

    # Projectors
    projector_hidden_size: int = 4096

    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False

    # Safety head
    num_safety_categories: int = 8

    # Model variant
    model_size: ModelSize = ModelSize.PRIMARY_7B

    # Loss weights (pretrain)
    loss_weight_text: float = 1.0
    loss_weight_vision: float = 0.3
    loss_weight_audio: float = 0.2
    loss_weight_alignment: float = 0.05

    @classmethod
    def from_size(cls, size: ModelSize) -> "LumenConfig":
        if size == ModelSize.DRAFT_1B:
            return cls(
                hidden_size=2048,
                num_hidden_layers=16,
                num_attention_heads=16,
                num_key_value_heads=4,
                head_dim=128,
                intermediate_size=7168,
                vision_hidden_size=384,
                vision_num_layers=6,
                vision_num_heads=6,
                vision_intermediate_size=1536,
                audio_hidden_size=256,
                audio_num_layers=6,
                audio_num_heads=4,
                projector_hidden_size=2048,
                model_size=size,
            )
        if size == ModelSize.EDGE_3B:
            return cls(
                hidden_size=3072,
                num_hidden_layers=24,
                num_attention_heads=24,
                num_key_value_heads=6,
                head_dim=128,
                intermediate_size=10752,
                vision_hidden_size=512,
                vision_num_layers=8,
                vision_num_heads=8,
                vision_intermediate_size=2048,
                audio_hidden_size=384,
                audio_num_layers=8,
                audio_num_heads=6,
                projector_hidden_size=3072,
                model_size=size,
            )
        return cls(model_size=size)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def is_global_attention_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.global_attention_layers
