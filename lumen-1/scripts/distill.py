"""Distill Lumen-1-7B into 3B or 1B student models."""

import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset

from lumen.model.config import ModelSize
from lumen.model.lumen_model import LumenForCausalLM
from lumen.tokenizer.lumen_tokenizer import LumenTokenizer
from lumen.train.distill import (
    LAYER_MAP_7B_TO_1B,
    LAYER_MAP_7B_TO_3B,
    Distiller,
    build_student,
)


class DistillationDataset(Dataset):
    def __init__(self, path: str, tokenizer: LumenTokenizer, max_length: int = 2048):
        import json

        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text = self.samples[idx].get("text", "")
        ids = self.tokenizer.encode(text, add_eos=True)[: self.max_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill Lumen-1")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-size", choices=["1b", "3b"], default="3b")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/lumen_tokenizer.model")
    parser.add_argument("--output", default="checkpoints/distilled")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    from lumen.model.config import LumenConfig

    teacher = LumenForCausalLM(LumenConfig.from_size(ModelSize.PRIMARY_7B))
    teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location="cpu"))

    student_size = ModelSize.EDGE_3B if args.student_size == "3b" else ModelSize.DRAFT_1B
    student = build_student(student_size)
    layer_map = LAYER_MAP_7B_TO_3B if args.student_size == "3b" else LAYER_MAP_7B_TO_1B

    tokenizer = (
        LumenTokenizer(args.tokenizer)
        if os.path.exists(args.tokenizer)
        else LumenTokenizer()
    )
    dataset = DistillationDataset(args.data, tokenizer)

    def collate(batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        ids = []
        for b in batch:
            pad = max_len - b["input_ids"].shape[0]
            ids.append(torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        return {"input_ids": torch.stack(ids)}

    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)
    distiller = Distiller(teacher, student, layer_map)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    distiller.train(loader, optimizer, max_steps=args.max_steps)

    os.makedirs(args.output, exist_ok=True)
    torch.save(student.state_dict(), f"{args.output}/lumen-1-{args.student_size}.pt")
    print(f"Saved student to {args.output}")


if __name__ == "__main__":
    main()
