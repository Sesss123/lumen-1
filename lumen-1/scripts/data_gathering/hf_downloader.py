import os
import json
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face datasets and convert to JSONL")
    parser.add_argument("--dataset", type=str, default="wikitext", help="HF Dataset name (e.g., wikitext, oscar)")
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train", help="Split to download (train, validation, test)")
    parser.add_argument("--output", type=str, default="../../data/text/hf_data.jsonl", help="Output JSONL path")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum samples to save")
    args = parser.parse_args()

    print(f"Downloading {args.dataset} ({args.config}) - {args.split} split...")
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Saving to {args.output}...")
    saved_count = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in dataset:
            if 'text' in item and item['text'].strip():
                # Format for Lumen-1 pretraining or alignment
                data_obj = {
                    "text": item['text'].strip()
                }
                f.write(json.dumps(data_obj) + '\n')
                saved_count += 1
                
            if saved_count >= args.max_samples:
                break
                
    print(f"Successfully saved {saved_count} samples to {args.output}")

if __name__ == "__main__":
    main()
