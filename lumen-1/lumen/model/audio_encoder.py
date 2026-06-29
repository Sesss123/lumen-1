"""Conformer-S audio encoder."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.model.config import LumenConfig


class ConvolutionSubsampling(nn.Module):
    """2x Conv2d subsampling for mel spectrograms."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) mel -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        # (B, C, T', F') -> (B, T', C)
        x = x.mean(dim=-1).transpose(1, 2)
        return self.norm(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return (x.transpose(1, 2) + residual)


class ConformerAttention(nn.Module):
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


class ConformerFeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        inner = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.SiLU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, conv_kernel_size: int = 31):
        super().__init__()
        self.ffn1 = ConformerFeedForward(dim)
        self.attn = ConformerAttention(dim, num_heads)
        self.conv = ConformerConvModule(dim, conv_kernel_size)
        self.ffn2 = ConformerFeedForward(dim)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_conv = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(self.norm_attn(x))
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ffn2(x)
        return x


class AudioEncoder(nn.Module):
    """Conformer-S encoder for log-mel spectrograms."""

    def __init__(self, config: LumenConfig):
        super().__init__()
        self.config = config
        self.subsample = ConvolutionSubsampling(1, config.audio_hidden_size)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    config.audio_hidden_size,
                    config.audio_num_heads,
                    config.audio_conv_kernel_size,
                )
                for _ in range(config.audio_num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.audio_hidden_size)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T, num_mel_bins) log-mel spectrogram
        Returns:
            (B, T', audio_hidden_size)
        """
        x = self.subsample(mel)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    @staticmethod
    def waveform_to_mel(
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
    ) -> torch.Tensor:
        """Convert waveform to log-mel (differentiable approximation)."""
        window = torch.hann_window(n_fft, device=waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        magnitude = spec.abs()
        mel_basis = torch.linspace(0, sample_rate / 2, n_mels, device=waveform.device)
        mel_spec = torch.einsum("bf,bft->bmt", mel_basis / (sample_rate / 2), magnitude)
        return torch.log(mel_spec.clamp(min=1e-10))
