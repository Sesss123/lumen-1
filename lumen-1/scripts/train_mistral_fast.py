import os
import json
import torch
import yaml
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None
from datasets import load_dataset

def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    # 1. Redact Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[REDACTED_EMAIL]', text)
    
    # 2. Redact Sri Lankan and International Phone Numbers
    phone_pattern = r'(?:\+94|0|0094)?\s*(?:[1-9]\d|7\d)\s*\d{3}\s*\d{4}\b|(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}\b'
    def phone_repl(match):
        m = match.group(0)
        digits = re.sub(r'\D', '', m)
        if len(digits) >= 8 and len(digits) <= 15:
            return '[REDACTED_PHONE]'
        return m
    text = re.sub(phone_pattern, phone_repl, text)
    
    # 3. Redact API Keys, Tokens, Passwords, etc.
    key_patterns = [
        (r'(?i)(api[-_]?key|secret[-_]?key|auth[-_]?token|password|passwd|bearer|token)(?:\s*[:=]\s*["\'\b]?)\s*([a-zA-Z0-9_\-\.~]{16,})["\'\b]?', r'\1: [REDACTED_KEY]'),
        (r'\b(?:xox[bpa]-[0-9]{12}-[0-9]{12}-[a-zA-Z0-9]{24}|AIzaSy[a-zA-Z0-9_\-]{33}|[a-zA-Z0-9]{40}|[a-zA-Z0-9]{32})\b', '[REDACTED_KEY]')
    ]
    for pattern, replacement in key_patterns:
        text = re.sub(pattern, replacement, text)
        
    return text

def sanitize_data_structure(data):
    if isinstance(data, str):
        return sanitize_text(data)
    elif isinstance(data, list):
        return [sanitize_data_structure(item) for item in data]
    elif isinstance(data, dict):
        return {key: sanitize_data_structure(val) for key, val in data.items()}
    return data

def sanitize_batch(batch):
    sanitized = {}
    for key, values in batch.items():
        sanitized[key] = [sanitize_data_structure(val) for val in values]
    return sanitized


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
CONFIG_PATH = "../configs/sft.yaml"
if not os.path.isabs(CONFIG_PATH):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.abspath(os.path.join(script_dir, CONFIG_PATH))

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def main():
    print(f"🚀 Starting Mistral-7B QLoRA SFT Training...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    # 2. BitsAndBytes 4-bit Quantization Config (BF16 auto-detection)
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    print(f"BF16 Support: {bf16_supported}. Using compute dtype: {compute_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )

    # 3. Load Model
    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype
    )
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    # model = get_peft_model(model, lora_config)

    # 5. Load Dataset (detects JSON/JSONL vs CSV format automatically)
    # (JSON, JSONL හෝ CSV format එක extension එකෙන් හඳුනාගෙන auto load කරයි)
    dataset_path = config["dataset_name"]
    file_ext = os.path.splitext(dataset_path)[1].lower()
    format_type = "csv" if file_ext == ".csv" else "json"
    
    print(f"Loading Dataset ({format_type.upper()}) from {dataset_path}...")
    dataset = load_dataset(format_type, data_files=dataset_path, split="train")

    # Sanitize the dataset for privacy and security
    print("Redacting any PII or credentials from dataset...")
    dataset = dataset.map(sanitize_batch, batched=True)

    # 6. Training Arguments
    import inspect
    fp16_val = not bf16_supported
    bf16_val = bf16_supported
    
    if SFTConfig is not None:
        # Inspect SFTConfig constructor parameters
        sig_config = inspect.signature(SFTConfig.__init__).parameters
        sft_args = {
            "output_dir": config["output_dir"],
            "per_device_train_batch_size": config["per_device_train_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "learning_rate": float(config["learning_rate"]),
            "logging_steps": config["logging_steps"],
            "num_train_epochs": config["num_train_epochs"],
            "save_steps": config["save_steps"],
            "optim": config["optim"],
            "fp16": fp16_val,
            "bf16": bf16_val
        }
        if "max_seq_length" in sig_config:
            sft_args["max_seq_length"] = config["max_seq_length"]
        if "dataset_text_field" in sig_config:
            sft_args["dataset_text_field"] = config.get("dataset_text_field", "messages")
            
        training_args = SFTConfig(**sft_args)
    else:
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=float(config["learning_rate"]),
            logging_steps=config["logging_steps"],
            num_train_epochs=config["num_train_epochs"],
            save_steps=config["save_steps"],
            optim=config["optim"],
            fp16=fp16_val,
            bf16=bf16_val
        )

    # Setup training log path
    log_dir = config["output_dir"]
    if not os.path.isabs(log_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.abspath(os.path.join(script_dir, log_dir))
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, "training_log.json")

    # 7. SFT Trainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": lora_config,
        "args": training_args,
        "callbacks": [DashboardCallback(log_filepath)]
    }
    
    # Check what parameters SFTTrainer accepts and route missing ones
    sig_trainer = inspect.signature(SFTTrainer.__init__).parameters
    
    if "processing_class" in sig_trainer:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig_trainer:
        trainer_kwargs["tokenizer"] = tokenizer
        
    if SFTConfig is not None:
        sig_config = inspect.signature(SFTConfig.__init__).parameters
        if "max_seq_length" not in sig_config and "max_seq_length" in sig_trainer:
            trainer_kwargs["max_seq_length"] = config["max_seq_length"]
        if "dataset_text_field" not in sig_config and "dataset_text_field" in sig_trainer:
            trainer_kwargs["dataset_text_field"] = config.get("dataset_text_field", "messages")
    else:
        if "max_seq_length" in sig_trainer:
            trainer_kwargs["max_seq_length"] = config["max_seq_length"]
        if "dataset_text_field" in sig_trainer:
            trainer_kwargs["dataset_text_field"] = config.get("dataset_text_field", "messages")
            
    trainer = SFTTrainer(**trainer_kwargs)

    # 8. Train
    print("Starting Training...")
    trainer.train()
    
    # 9. Save
    print(f"Saving Model to {config['output_dir']}...")
    trainer.model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print("✅ Training Complete!")

if __name__ == "__main__":
    main()
