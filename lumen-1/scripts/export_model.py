import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PEFT_MODEL = "../checkpoints/lumen_mistral_finetuned"
MERGED_OUTPUT = "../checkpoints/lumen_mistral_merged"

def export_to_safetensors(base_model_path, adapter_path, output_path):
    # Load base model (Base model එක load කරගන්නවා)
    print(f"Loading Base Model: {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cpu", # Merge on CPU if GPU RAM is low
    )
    
    # Load Tokenizer (Tokenizer එක load කරගන්නවා)
    print(f"Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Merge PEFT adapter weights (LoRA adapter weights base model එක සමඟ එකතු කරනවා)
    print(f"Loading and Merging LoRA weights from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    
    # Save the merged model in Safetensors format (එකතු කරන ලද model එක safe-serialization සමඟ save කරයි)
    print(f"Saving merged model to {output_path} (Safetensors format)...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Model successfully merged and exported!")
    print(f"Merged model path: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge PEFT adapter with base model and export to Safetensors")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model identifier or path")
    parser.add_argument("--adapter", default="../checkpoints/lumen_mistral_finetuned", help="PEFT adapter directory path")
    parser.add_argument("--output", default="../checkpoints/lumen_mistral_merged", help="Output directory path for merged model")
    args = parser.parse_args()
    
    export_to_safetensors(args.base_model, args.adapter, args.output)

if __name__ == "__main__":
    main()
