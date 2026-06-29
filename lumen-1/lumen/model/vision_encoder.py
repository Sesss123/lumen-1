"""ViT-SigLIP vision encoder."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.model.config import LumenConfig


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1
        )
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.proj(out)


class VisionMLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VisionSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = VisionMLP(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-B/16 style encoder with SigLIP-style LayerNorm (no class token)."""

    def __init__(self, config: LumenConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(
            config.vision_image_size,
            config.vision_patch_size,
            3,
            config.vision_hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                VisionEncoderLayer(
                    config.vision_hidden_size,
                    config.vision_num_heads,
                    config.vision_intermediate_size,
                )
                for _ in range(config.vision_num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.vision_hidden_size)

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) normalized images
        Returns:
            (B, num_patches, vision_hidden_size)
        """
        x = self.patch_embed(pixel_values)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def encode_video(self, frames: torch.Tensor, max_frames: int = 32) -> torch.Tensor:
        """Encode video frames sampled at 1 fps. frames: (B, T, C, H, W)."""
        bsz, num_frames = frames.shape[:2]
        if num_frames > max_frames:
            indices = torch.linspace(0, num_frames - 1, max_frames).long()
            frames = frames[:, indices]
        flat = frames.reshape(-1, *frames.shape[2:])
        encoded = self.forward(flat)
        patches_per_frame = encoded.shape[1]
        encoded = encoded.view(bsz, -1, patches_per_frame, self.config.vision_hidden_size)
        return encoded.reshape(bsz, -1, self.config.vision_hidden_size)
