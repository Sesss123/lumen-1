"""Stage 1-2 pretraining: text then multimodal."""

import argparse

import torch
from torch.utils.data import DataLoader

from lumen.data.dataset import LumenDataset
from lumen.model.lumen_model import LumenForCausalLM
from lumen.tokenizer.lumen_tokenizer import LumenTokenizer
from lumen.train.trainer import LumenTrainer, load_config


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids, labels = [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lumen-1 pretraining")
    parser.add_argument("--config", default="configs/lumen-7b.yaml")
    parser.add_argument("--data", required=True, help="Training data path")
    parser.add_argument("--stage", choices=["text", "multimodal"], default="text")
    parser.add_argument("--output-dir", default="checkpoints/pretrain")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage == "multimodal":
        config.use_gradient_checkpointing = True

    lr = args.lr or (3e-4 if args.stage == "text" else 1e-4)
    model = LumenForCausalLM(config)

    tokenizer_path = "tokenizer/lumen_tokenizer.model"
    tokenizer = LumenTokenizer(tokenizer_path) if __import__("os").path.exists(tokenizer_path) else LumenTokenizer()

    dataset = LumenDataset(args.data, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    trainer = LumenTrainer(
        model=model,
        config=config,
        train_dataloader=loader,
        learning_rate=lr,
        max_steps=args.max_steps,
        schedule="wsd" if args.stage == "text" else "cosine",
        output_dir=args.output_dir,
        use_fsdp=args.fsdp,
        gradient_accumulation_steps=8,
    )

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state)

    print(f"Starting {args.stage} pretraining: {model.count_parameters()}")
    trainer.train()


if __name__ == "__main__":
    main()
