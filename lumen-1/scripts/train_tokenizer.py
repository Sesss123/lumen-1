"""Train LumenTokenizer v1 on a text corpus."""

import argparse
import tempfile
from pathlib import Path

from lumen.tokenizer.lumen_tokenizer import LumenTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LumenTokenizer v1")
    parser.add_argument("--corpus", nargs="+", help="Corpus text files")
    parser.add_argument("--output-dir", default="tokenizer", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=128256)
    args = parser.parse_args()

    corpus_paths = args.corpus
    if not corpus_paths:
        # Bootstrap with minimal corpus for development
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        tmp.write(
            "Lumen-1 is a local-first multimodal foundation model.\n"
            "<|system|> You are a helpful assistant.\n"
            "<|user|> Describe this image. <image> </image>\n"
            "<|assistant|> The image shows a landscape.\n"
        )
        tmp.close()
        corpus_paths = [tmp.name]

    tokenizer = LumenTokenizer.train(
        corpus_paths=corpus_paths,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )
    print(f"Tokenizer trained: vocab_size={tokenizer.vocab_size}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
