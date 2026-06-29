# Lumen-1 Architecture

## Overview

Lumen-1 is a dense hybrid multimodal transformer with a decoder-only text backbone and separate modality encoders projecting into a unified 4096-dim token space.

## Decoder (6.1B params)

- 32 layers, hidden dim 4096
- SwiGLU FFN (intermediate 14336)
- Pre-RMSNorm
- RoPE (θ=1M) with YaRN scaling
- GQA: 32 Q heads, 8 KV heads

## HLG Attention

Alternating attention modes every 4 layers:

- **SWA** (layers 0–3, 5–7, …): sliding window 4096
- **GSA** (layers 4, 8, …): global sink tokens + sparse stride-4096 attention

## Vision Encoder (0.45B)

- ViT-B/16, 384×384 input
- 12 layers, 768 dim
- 2-layer MLP projector → 4096 dim

## Audio Encoder (0.35B)

- Conformer-S, 12 layers, 512 dim
- 16 kHz log-mel (80 bins)
- 50 tokens/second after projection

## Safety Head (0.05B)

- 2-layer MLP on last hidden state
- 8 harm categories

## Inference Optimizations

- FlashAttention-2 (SWA layers)
- KV cache INT8/FP16
- Speculative decoding with 1B draft
- Encoder embedding cache
- Prefix caching for system prompts

## Quantization

AWQ/GPTQ INT4 for local; FP8 for H100 cloud serving; GGUF for llama.cpp.
