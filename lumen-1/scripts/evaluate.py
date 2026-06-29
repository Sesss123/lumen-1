"""Run Lumen-1 evaluation benchmarks."""

import argparse
import json

import torch

from lumen.eval.harness import EvalHarness
from lumen.model.lumen_model import LumenForCausalLM
from lumen.tokenizer.lumen_tokenizer import LumenTokenizer
from lumen.train.trainer import load_config


def load_benchmark(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lumen-1")
    parser.add_argument("--config", default="configs/lumen-7b.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/lumen_tokenizer.model")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu", "gsm8k", "truthfulqa", "toxigen"])
    parser.add_argument("--data-dir", default="eval_data")
    args = parser.parse_args()

    config = load_config(args.config)
    model = LumenForCausalLM(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    tokenizer = LumenTokenizer(args.tokenizer) if __import__("os").path.exists(args.tokenizer) else LumenTokenizer()

    harness = EvalHarness(model, tokenizer)
    benchmarks = {}
    for name in args.benchmarks:
        path = f"{args.data_dir}/{name}.jsonl"
        try:
            benchmarks[name] = load_benchmark(path)
        except FileNotFoundError:
            print(f"Skipping {name}: {path} not found")

    results = harness.run_all(benchmarks)
    harness.print_report(results)


if __name__ == "__main__":
    main()
