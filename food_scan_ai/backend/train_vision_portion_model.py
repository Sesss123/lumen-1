# =====================================================================
# Google Colab Training Script: AI Vision & Portion Size Estimation
# =====================================================================
# මෙම ස්ක්‍රිප්ට් එක Google Colab Free Tier (T4 GPU) මත ධාවනය කර,
# කෑම ප්‍රමාණයන් සහ Thinking Knowledge සහිත AI මාදිලිය පුහුණු කළ හැක.
# =====================================================================

import os
import torch

def setup_colab_environment():
    print("🛠️ Installing Unsloth, TRL, and dependencies for fast 4-bit fine-tuning...")
    os.system("pip install --upgrade pip")
    os.system("pip install unsloth_zoo")
    os.system("pip install --no-deps \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
    os.system("pip install --no-deps xformers trl peft accelerate bitsandbytes")
    print("✅ Environment installation completed!")

def train_model():
    try:
        from unsloth import FastLanguageModel # type: ignore
        from datasets import load_dataset # type: ignore
        from trl import SFTTrainer # type: ignore
        from transformers import TrainingArguments # type: ignore
        from unsloth.chat_templates import get_chat_template # type: ignore
    except ImportError:
        print("⚠️ Unsloth not detected. Running auto-install...")
        setup_colab_environment()
        from unsloth import FastLanguageModel # type: ignore
        from datasets import load_dataset # type: ignore
        from trl import SFTTrainer # type: ignore
        from transformers import TrainingArguments # type: ignore
        from unsloth.chat_templates import get_chat_template # type: ignore

    max_seq_length = 2048
    dtype = None # Auto detection
    load_in_4bit = True # Use 4bit quantization to fit in Colab T4 GPU

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"🚀 Loading base model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )

    # Do model patching for LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Target Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    dataset_path = "vision_portion_sft_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset '{dataset_path}' not found! Run generate_vision_portion_sft.py first.")
        return

    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Fast training (~15 mins on Colab)
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            save_strategy = "no",
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    print("🔥 Starting AI Supervised Fine-Tuning (SFT)...")
    trainer_stats = trainer.train()
    print(f"✅ Training completed in {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes!")

    output_model_dir = "vision_portion_lora_model"
    print(f"💾 Saving trained LoRA weights to '{output_model_dir}'...")
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("🎉 All done! Model is ready for inference.")

if __name__ == "__main__":
    print("=====================================================================")
    print("🌟 Antigravity Vision Portion Estimation AI Trainer Started 🌟")
    print("=====================================================================")
    train_model()
