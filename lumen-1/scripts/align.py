import os
import json
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from peft import PeftModel, get_peft_model, LoraConfig
from trl import DPOTrainer
from datasets import load_dataset

class DashboardCallback(TrainerCallback):
    """
    Custom callback to log training metrics (Loss, LR, Epoch, Step) to a JSON file
    in real-time, allowing the FastAPI dashboard to parse and plot the metrics.
    """
    def __init__(self, log_filepath="training_log.json"):
        super().__init__()
        self.log_filepath = log_filepath
        self.history = []
        # Clear existing logs
        if os.path.exists(self.log_filepath):
            try:
                os.remove(self.log_filepath)
            except Exception:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Clean up the logs to only contain serializable/simple types
            clean_logs = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    clean_logs[k] = v
                elif hasattr(v, "item"): # PyTorch tensor
                    clean_logs[k] = v.item()
                else:
                    clean_logs[k] = str(v)

            log_entry = {
                "step": state.global_step,
                "epoch": round(state.epoch or 0, 4),
                "max_steps": state.max_steps,
                **clean_logs
            }
            self.history.append(log_entry)
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_filepath)), exist_ok=True)
                with open(self.log_filepath, "w") as f:
                    json.dump(self.history, f, indent=2)
            except Exception as e:
                print(f"Error writing callback logs: {e}")

# Configuration
CONFIG_PATH = "../configs/dpo.yaml"
if not os.path.isabs(CONFIG_PATH):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.abspath(os.path.join(script_dir, CONFIG_PATH))

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def main():
    print("🚀 Starting Model Alignment (DPO)...")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model first (ප්‍රධාන base model එක මුලින්ම load කරගන්නවා)
    print("Loading Base Model for DPO...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load PEFT adapter on top of the base model (Base model එක මතට SFT adapter එක load කරගන්නවා)
    print(f"Loading SFT adapters from {config['model_name_or_path']}...")
    model = PeftModel.from_pretrained(base_model, config["model_name_or_path"], is_trainable=True)
    
    # In DPO we need a reference model (usually the SFT model). 
    # DPOTrainer can automatically create it if we pass the Peft model.
    
    # Load Preference Dataset (detects JSON/JSONL vs CSV format automatically)
    # (JSON, JSONL හෝ CSV format එක extension එකෙන් හඳුනාගෙන auto load කරයි)
    dataset_path = config["dataset_name"]
    file_ext = os.path.splitext(dataset_path)[1].lower()
    format_type = "csv" if file_ext == ".csv" else "json"
    
    print(f"Loading Preference Dataset ({format_type.upper()}) from {dataset_path}...")
    dataset = load_dataset(format_type, data_files=dataset_path, split="train")
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        logging_steps=config["logging_steps"],
        num_train_epochs=config["num_train_epochs"],
        fp16=True,
        remove_unused_columns=False
    )
    
    # Setup training log path
    log_dir = config["output_dir"]
    if not os.path.isabs(log_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.abspath(os.path.join(script_dir, log_dir))
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, "training_log.json")

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # None defaults to turning off adapters if using PEFT
        args=training_args,
        beta=config["beta"],
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[DashboardCallback(log_filepath)]
    )
    
    print("Starting DPO Training...")
    trainer.train()
    
    print(f"Saving Aligned Model to {config['output_dir']}...")
    trainer.save_model(config["output_dir"])
    print("✅ Alignment Complete!")

if __name__ == "__main__":
    main()
