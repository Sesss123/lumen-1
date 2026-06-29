#!/usr/bin/env python3
"""
Lumen-1 Model Quantization Script.
Performs AWQ/GPTQ weight-only simulation or calls AutoAWQ/AutoGPTQ.
"""

import os
import sys
import argparse
import torch

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from lumen.model.lumen_model import LumenForCausalLM
from lumen.model.config import LumenConfig, ModelSize
from lumen.inference.quantize import quantize_awq, quantize_gptq


def main():
    parser = argparse.ArgumentParser(description="Quantize Lumen-1 model weights to 4-bit/8-bit")
    parser.add_argument("--model_size", type=str, default="1b", choices=["1b", "3b", "7b"], help="Model size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Input model checkpoint path (.pt)")
    parser.add_argument("--output_path", type=str, default="../checkpoints/lumen_quantized.pt", help="Path to save quantized model")
    parser.add_argument("--method", type=str, default="awq", choices=["awq", "gptq"], help="Quantization method")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Bits per weight channel")
    args = parser.parse_args()

    # Resolve output path
    out_path = args.output_path
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(os.path.join(script_dir, out_path))

    print(f"🤖 Initializing model structure for size: Lumen-1-{args.model_size}...")
    config = LumenConfig.from_size(ModelSize(args.model_size))
    model = LumenForCausalLM(config)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📂 Loading checkpoints from {args.checkpoint}...")
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}. Using uninitialized weights.")
    else:
        print("💡 No input checkpoint provided. Quantizing model structure using initialization weights.")

    # Create dummy calibration data (mock token sequences)
    print("📊 Generating calibration data...")
    calibration_data = []
    # Create 8 random sequences of length 64
    for _ in range(8):
        input_ids = torch.randint(100, 10000, (1, 64))
        calibration_data.append({"input_ids": input_ids})

    print(f"⚡ Starting {args.method.upper()} quantization ({args.bits}-bit)...")
    if args.method == "awq":
        quant_model = quantize_awq(
            model=model,
            calibration_data=calibration_data,
            bits=args.bits,
            output_path=out_path
        )
    else:
        quant_model = quantize_gptq(
            model=model,
            calibration_data=calibration_data,
            bits=args.bits,
            output_path=out_path
        )

    print(f"✅ Quantization successfully finished!")
    print(f"💾 Quantized weights stored at: {out_path}")


if __name__ == "__main__":
    main()
