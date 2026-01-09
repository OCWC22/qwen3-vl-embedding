#!/usr/bin/env python3
"""
MINIMAL SFT (Supervised Fine-Tuning) for Qwen3-VL Base Model
============================================================

Fine-tune the generation model to output correct actions.
Works with both Dense (32B) and MoE (A30B-A3B) models.

Requirements:
    pip install transformers peft bitsandbytes accelerate torch pillow trl

Run:
    python train_sft.py

GPU Memory:
    - Qwen3-VL-3B: ~12GB
    - Qwen3-VL-8B: ~20GB
    - Qwen3-VL-A3B (MoE): ~24GB
    - Qwen3-VL-32B: ~40GB (need A100/H100)

Time: ~4-8 hours for 100 examples, 3 epochs
"""

import os
import json
import torch
from pathlib import Path

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
from PIL import Image

# =============================================================================
# CONFIG
# =============================================================================

# Choose your model:
# MODEL_NAME = "Qwen/Qwen3-VL-3B"           # Smallest, fastest
# MODEL_NAME = "Qwen/Qwen3-VL-8B"           # Good balance
MODEL_NAME = "Qwen/Qwen3-VL-A3B"            # MoE - best quality/speed tradeoff
# MODEL_NAME = "Qwen/Qwen3-VL-32B"          # Largest, needs big GPU

DATA_PATH = "data/sft_train.jsonl"
OUTPUT_DIR = "outputs/sft_higgsfield"
SCREENSHOT_DIR = "data/screenshots"

# Training params
EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUM = 4  # Effective batch = 4
LEARNING_RATE = 2e-4
LORA_R = 32
LORA_ALPHA = 64
MAX_LENGTH = 1024


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_sft_data(data_path):
    """Load SFT data and convert to HuggingFace Dataset format."""
    with open(data_path) as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(raw_data)} SFT examples")

    # Convert to simple format for SFTTrainer
    processed = []
    for item in raw_data:
        messages = item["messages"]

        # Build conversation string
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                text_parts.append(f"<|{role}|>\n{content}")
            elif isinstance(content, list):
                # Handle multimodal content
                text = ""
                for c in content:
                    if c["type"] == "text":
                        text += c["text"]
                    elif c["type"] == "image":
                        text += "<image>"
                text_parts.append(f"<|{role}|>\n{text}")

        full_text = "\n".join(text_parts) + "<|endoftext|>"
        processed.append({"text": full_text})

    return Dataset.from_list(processed)


# =============================================================================
# TRAINING
# =============================================================================

def main():
    print("=" * 60)
    print("SFT FINE-TUNING FOR COMPUTER USE AGENT")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)

    # LoRA config - target all important modules
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    dataset = load_sft_data(DATA_PATH)
    print(f"Dataset size: {len(dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",  # or "tensorboard"
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=processor.tokenizer,
        max_seq_length=MAX_LENGTH,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)

    print("SFT training complete!")


if __name__ == "__main__":
    main()
