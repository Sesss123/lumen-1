#!/usr/bin/env python3
"""
Lumen-1 Speculative Decoding Demo.
Compares decoding speed of Lumen-1-7B (Target) accelerated by Lumen-1-1B (Draft)
against standard autoregressive decoding.
"""

import os
import sys
import argparse
import time
import torch

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from lumen.model.lumen_model import LumenForCausalLM
from lumen.model.config import LumenConfig, ModelSize
from lumen.tokenizer.lumen_tokenizer import LumenTokenizer
from lumen.inference.speculative import SpeculativeDecoder


def main():
    parser = argparse.ArgumentParser(description="Lumen-1 Speculative Decoding Speed Demo")
    parser.add_argument("--prompt", type=str, default="Tell me a beautiful destination in Sri Lanka in Sinhala.", help="Inference prompt")
    parser.add_argument("--target_size", type=str, default="3b", choices=["3b", "7b"], help="Target model size")
    parser.add_argument("--draft_size", type=str, default="1b", choices=["1b"], help="Draft model size")
    parser.add_argument("--gamma", type=int, default=5, help="Number of speculative steps to draft before target check")
    parser.add_argument("--tokens_to_generate", type=int, default=32, help="Number of new tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Running on device: {device}")

    # Load Tokenizer
    tokenizer_path = os.path.join(project_root, "tokenizer", "lumen_tokenizer.model")
    if os.path.exists(tokenizer_path):
        print(f"📖 Loading tokenizer from {tokenizer_path}...")
        tokenizer = LumenTokenizer(tokenizer_path)
    else:
        print("⚠️ Tokenizer model not found at standard path. Initializing placeholder tokenizer.")
        tokenizer = LumenTokenizer()

    # Load Models
    print(f"🤖 Initializing Target Model (Lumen-1-{args.target_size})...")
    target_config = LumenConfig.from_size(ModelSize(args.target_size))
    target_model = LumenForCausalLM(target_config)

    print(f"🤖 Initializing Draft Model (Lumen-1-{args.draft_size})...")
    draft_config = LumenConfig.from_size(ModelSize(args.draft_size))
    draft_model = LumenForCausalLM(draft_config)

    # Move to device and put in eval mode
    target_model.to(device).eval()
    draft_model.to(device).eval()

    # Tokenize input prompt
    input_ids = tokenizer.encode(args.prompt)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"\n📝 User Prompt: '{args.prompt}'")
    print(f"🧬 Starting Speculative Decoding vs Standard Autoregressive comparison...")

    # --- Method 1: Speculative Decoding ---
    print("\n⚡ Running Speculative Decoding...")
    spec_decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        gamma=args.gamma,
        device=device
    )

    t_start = time.time()
    current_ids = input_ids_tensor.clone()
    
    # We generate tokens step-by-step
    generated_count = 0
    while generated_count < args.tokens_to_generate:
        prev_len = current_ids.shape[1]
        current_ids = spec_decoder.decode_step(current_ids)
        new_tokens_added = current_ids.shape[1] - prev_len
        generated_count += new_tokens_added
        
    t_end = time.time()
    spec_time = t_end - t_start
    spec_tokens_per_sec = generated_count / spec_time
    
    spec_text = tokenizer.decode(current_ids[0][input_ids_tensor.shape[1]:].tolist())

    # --- Method 2: Standard Autoregressive ---
    print("\n⏳ Running Standard Autoregressive Decoding...")
    t_start_std = time.time()
    with torch.no_grad():
        std_generated = target_model.generate(
            input_ids=input_ids_tensor,
            max_new_tokens=args.tokens_to_generate,
            temperature=1.0,
            eos_token_id=tokenizer.token_to_id(tokenizer.EOS_TOKEN)
        )
    t_end_std = time.time()
    std_time = t_end_std - t_start_std
    std_generated_count = std_generated.shape[1] - input_ids_tensor.shape[1]
    std_tokens_per_sec = std_generated_count / std_time

    std_text = tokenizer.decode(std_generated[0][input_ids_tensor.shape[1]:].tolist())

    # --- Performance Summary ---
    print("\n✨=== INFERENCE SPEED PERFORMANCE ===✨")
    print(f"🚀 Speculative Decoding Time: {spec_time:.4f}s ({spec_tokens_per_sec:.2f} tok/s)")
    print(f"   Draft Acceptance Rate:     {spec_decoder.acceptance_rate * 100:.1f}%")
    print(f"🐢 Standard Decoding Time:    {std_time:.4f}s ({std_tokens_per_sec:.2f} tok/s)")
    speedup = (spec_tokens_per_sec / std_tokens_per_sec) if std_tokens_per_sec > 0 else 1.0
    print(f"🏆 Speedup Multiplier:        {speedup:.2f}x faster")
    print("=======================================")

    print("\n📋 Generated Sample (Speculative):")
    print(f"'{spec_text.strip()}'")


if __name__ == "__main__":
    main()
