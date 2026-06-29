import os
import sys
import argparse
import subprocess
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_export_to_gguf(base_model_path, adapter_path, output_gguf_path):
    # Determine temporary HF merge directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_merge_dir = os.path.join(script_dir, "..", "checkpoints", "lumen_temp_hf_merge")
    os.makedirs(temp_merge_dir, exist_ok=True)
    
    try:
        # Step 1: Merge weights and export to HF Safetensors
        print(f"🔄 Phase 1: Merging LoRA weights from {adapter_path} into base model {base_model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        print("📂 Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        
        print(f"💾 Saving temporary HuggingFace weights to {temp_merge_dir}...")
        model.save_pretrained(temp_merge_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_merge_dir)
        
        # Free memory
        del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Step 2: Download convert_hf_to_gguf.py if not present
        convert_script = os.path.join(script_dir, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script):
            print("🌐 convert_hf_to_gguf.py not found. Downloading from llama.cpp repository...")
            import urllib.request
            url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
            try:
                urllib.request.urlretrieve(url, convert_script)
                print("✅ Downloaded convert_hf_to_gguf.py successfully.")
            except Exception as e:
                print(f"❌ Failed to download script: {e}. Attempting direct pip library execution...")

        # Step 3: Install gguf package
        try:
            import gguf
            print("📦 'gguf' package is already installed.")
        except ImportError:
            print("📦 Installing 'gguf' dependency via pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "gguf"], check=True)
            
        # Step 4: Run llama.cpp conversion script
        print(f"⚡ Phase 2: Converting HuggingFace format to GGUF (4-bit Q4_0 format)...")
        
        # Ensure output filename ends with .gguf
        if not output_gguf_path.endswith(".gguf"):
            output_gguf_path += ".gguf"
            
        cmd = [
            sys.executable,
            convert_script,
            temp_merge_dir,
            "--outfile", output_gguf_path,
            "--outtype", "q4_0"
        ]
        
        print(f"💻 Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"🎉 Success! Quantized GGUF model created at: {output_gguf_path}")
        else:
            raise RuntimeError(f"llama.cpp conversion failed with exit code {result.returncode}")
            
    finally:
        # Step 5: Clean up temporary HF directory
        if os.path.exists(temp_merge_dir):
            print("🧹 Cleaning up temporary merge files...")
            try:
                shutil.rmtree(temp_merge_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temp folder: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and quantize to GGUF format")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model path")
    parser.add_argument("--adapter", default="../checkpoints/lumen_mistral_finetuned", help="Adapter path")
    parser.add_argument("--output", default="../checkpoints/lumen_mistral_merged.gguf", help="Output GGUF filepath")
    args = parser.parse_args()
    
    merge_and_export_to_gguf(args.base_model, args.adapter, args.output)

if __name__ == "__main__":
    main()
