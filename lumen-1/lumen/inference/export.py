"""Model export: Safetensors, GGUF, ONNX."""

import json
from pathlib import Path
from typing import Optional

import torch

from lumen.model.lumen_model import LumenForCausalLM


def export_safetensors(model: LumenForCausalLM, output_dir: str) -> str:
    """Export model weights to HuggingFace-compatible safetensors."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
        return str(output_dir / "pytorch_model.bin")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    save_file(state, output_dir / "model.safetensors")

    config = {
        "architectures": ["LumenForCausalLM"],
        "model_type": "lumen",
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return str(output_dir)


def export_gguf(model: LumenForCausalLM, output_path: str, quantize: str = "Q4_K_M") -> str:
    """
    Export to GGUF format for llama.cpp.
    Writes metadata + raw tensors; full conversion requires llama.cpp quantize tool.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "general.architecture": "lumen",
        "lumen.context_length": model.config.max_position_embeddings,
        "lumen.embedding_length": model.config.hidden_size,
        "lumen.block_count": model.config.num_hidden_layers,
        "lumen.attention.head_count": model.config.num_attention_heads,
        "lumen.attention.head_count_kv": model.config.num_key_value_heads,
        "lumen.vocab_size": model.config.vocab_size,
        "quantize": quantize,
    }

    # Save intermediate PyTorch checkpoint for llama.cpp conversion script
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "quantize_target": quantize,
    }
    torch.save(checkpoint, output.with_suffix(".pt"))
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return str(output)


def export_onnx(model: LumenForCausalLM, output_dir: str, opset: int = 17) -> str:
    """Export vision and audio encoder subgraphs to ONNX."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = "cpu"

    # Vision encoder
    dummy_image = torch.randn(1, 3, model.config.vision_image_size, model.config.vision_image_size)
    torch.onnx.export(
        model.model.vision_encoder,
        dummy_image,
        output_dir / "vision_encoder.onnx",
        input_names=["pixel_values"],
        output_names=["vision_features"],
        dynamic_axes={"pixel_values": {0: "batch"}, "vision_features": {0: "batch", 1: "seq"}},
        opset_version=opset,
    )

    # Audio encoder
    dummy_mel = torch.randn(1, 3000, model.config.audio_num_mel_bins)
    torch.onnx.export(
        model.model.audio_encoder,
        dummy_mel,
        output_dir / "audio_encoder.onnx",
        input_names=["mel_spectrogram"],
        output_names=["audio_features"],
        dynamic_axes={"mel_spectrogram": {0: "batch", 1: "time"}},
        opset_version=opset,
    )

    return str(output_dir)
