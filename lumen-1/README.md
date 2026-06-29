# Lumen-1

**Lumen-1** (Light Unified Multimodal Encoder Network) is a local-first, 7.2B-parameter dense multimodal foundation model supporting text, vision, and audio.

## Architecture

- **Decoder**: 32-layer transformer with Hybrid Local-Global (HLG) attention and GQA
- **Vision**: ViT-SigLIP encoder (384×384, patch 16)
- **Audio**: Conformer-S encoder (16 kHz, 80 mel bins)
- **Safety**: 6-layer safety stack + in-model safety head

## Model Family

| Model | Params | Use Case |
|-------|--------|----------|
| Lumen-1-7B | ~7.2B | Primary local/cloud model |
| Lumen-1-3B | ~3B | Edge deployment (distilled) |
| Lumen-1-1B | ~1B | Speculative decoding draft |

## Quick Start

```bash
pip install -e .
pip install -e ".[train,eval,dev]"

# Train tokenizer
python scripts/train_tokenizer.py --output-dir tokenizer

# Pretrain (text stage)
python scripts/pretrain.py --config configs/lumen-7b.yaml --data data/text --stage text

# Alignment (SFT / DPO / safety)
python scripts/align.py --mode sft --data data/sft.jsonl

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/final/model.pt --benchmarks mmlu

# Export
python scripts/export_model.py --checkpoint checkpoints/final/model.pt --format safetensors --output exports/

# Distill 7B -> 3B/1B
python scripts/distill.py --teacher-checkpoint teacher.pt --student-size 3b --data data/text.jsonl
```

## Project Structure

```
lumen-1/
├── lumen/
│   ├── model/       # Architecture
│   ├── tokenizer/   # LumenTokenizer v1
│   ├── data/        # Data pipeline
│   ├── train/       # Training loops
│   ├── eval/        # Benchmark harness
│   ├── inference/   # Export, quantize, speculative decoding
│   └── safety/      # 6-layer safety stack
├── configs/         # YAML hyperparameters
├── scripts/         # CLI entry points
├── tests/           # Unit tests
└── docs/            # Model card & architecture
```

## Hardware (Inference)

| Config | VRAM |
|--------|------|
| FP16 | ~16 GB |
| INT4 (Q4_K_M) | ~5.5 GB |

RTX 3060 12GB (INT4) minimum; RTX 4090 24GB recommended.

## License

Research preview. See MODEL_CARD.md for usage terms.
