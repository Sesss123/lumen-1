# Lumen-1 Model Card

## Model Details

- **Name**: Lumen-1
- **Version**: 0.1.0 (research preview)
- **Type**: Multimodal causal language model (text + vision + audio)
- **Parameters**: ~7.2B (dense)
- **Context**: 32,768 tokens (128K via YaRN extrapolation)
- **License**: TBD

## Intended Use

- Local and cloud deployment for multimodal understanding
- Image captioning, visual QA, speech understanding, general chat
- Fine-tuning via LoRA/QLoRA on consumer hardware

## Out-of-Scope

- Not intended for medical, legal, or safety-critical decisions without human review
- Not a music generation model
- Not designed for real-time video generation

## Training Data

| Category | Volume |
|----------|--------|
| Text | 2T tokens |
| Image-text | 5B pairs |
| Audio | 100K hours |
| Instruction | 10M examples |

## Evaluation Targets

| Benchmark | Target |
|-----------|--------|
| MMLU | ≥ 65% |
| MMMU | ≥ 45% |
| AIR-Bench | ≥ 40% |
| TruthfulQA | ≥ 55% |

## Safety

6-layer safety stack: prompt processing, input filter, in-model safety head, output filter, alignment verifier, audit logger.

## Hardware Requirements

- **Training**: 64–128× H100 80GB
- **Inference**: 12 GB VRAM (INT4), 24 GB recommended (FP16)

## Limitations

- Smaller than frontier closed models (GPT-4o, Claude 3.5)
- Multimodal performance depends on training data quality
- Long-context beyond 32K is extrapolated, not fully trained
