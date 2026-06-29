"""Preprocess raw data into filtered Parquet/JSONL shards."""

import argparse
import json
from pathlib import Path

from lumen.data.filters import DedupFilter, PIIFilter, QualityFilter, SafetyFilter


def process_text_file(input_path: str, output_path: str) -> int:
    dedup = DedupFilter()
    quality = QualityFilter()
    safety = SafetyFilter()
    pii = PIIFilter()

    count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            text = line.strip()
            if not text:
                continue
            if dedup.is_duplicate(text):
                continue
            if not quality.passes(text):
                continue
            if not safety.passes(text):
                continue
            text = pii.redact(text)
            fout.write(json.dumps({"text": text}) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    n = process_text_file(args.input, args.output)
    print(f"Wrote {n} records to {args.output}")


if __name__ == "__main__":
    main()
