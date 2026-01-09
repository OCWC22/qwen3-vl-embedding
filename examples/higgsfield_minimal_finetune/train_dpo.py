#!/usr/bin/env python3
"""
MINIMAL DPO (Direct Preference Optimization) Training
=====================================================

DPO is a simpler alternative to PPO/RLHF.
Instead of online RL, it learns from preference pairs (chosen vs rejected).

Why DPO over PPO:
- No reward model needed
- No online environment interaction
- More stable training
- Same or better results

Requirements:
    pip install transformers peft bitsandbytes accelerate torch trl

Run:
    python train_dpo.py

GPU Memory: ~24-40GB depending on model
Time: ~2-4 hours for 50 preference pairs
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
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# =============================================================================
# CONFIG
# =============================================================================

# Use the SFT model as base, or start fresh
MODEL_NAME = "outputs/sft_higgsfield"  # Your SFT model
# MODEL_NAME = "Qwen/Qwen3-VL-A3B"     # Or start from base

DATA_PATH = "data/dpo_train.jsonl"
OUTPUT_DIR = "outputs/dpo_higgsfield"

EPOCHS = 1  # DPO usually needs fewer epochs
BATCH_SIZE = 1
GRADIENT_ACCUM = 4
LEARNING_RATE = 5e-5  # Lower than SFT
LORA_R = 16  # Can be smaller for DPO
LORA_ALPHA = 32
MAX_LENGTH = 1024
BETA = 0.1  # DPO temperature (lower = more aggressive)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_dpo_data(data_path):
    """
    Load DPO preference data.

    Expected format:
    {
        "prompt": {"system": "...", "user_image": "...", "user_text": "..."},
        "chosen": "response that should be preferred",
        "rejected": "response that should be avoided"
    }
    """
    with open(data_path) as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(raw_data)} preference pairs")

    processed = []
    for item in raw_data:
        prompt_parts = item["prompt"]

        # Build prompt string
        prompt = f"""<|system|>
{prompt_parts.get('system', 'You are a helpful assistant.')}
<|user|>
{prompt_parts.get('user_text', '')}
<|assistant|>
"""

        processed.append({
            "prompt": prompt,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })

    return Dataset.from_list(processed)


# =============================================================================
# TRAINING
# =============================================================================

def main():
    print("=" * 60)
    print("DPO TRAINING (Preference Optimization)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Beta: {BETA}")

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-A3B", trust_remote_code=True)

    # Quantization
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

    # Load reference model (frozen copy for DPO)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for LoRA (only on main model)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    dataset = load_dpo_data(DATA_PATH)
    print(f"Dataset size: {len(dataset)}")

    # DPO Config
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_LENGTH // 2,
    )

    # Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
    )

    # Train
    print("\nStarting DPO training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)

    print("DPO training complete!")


# =============================================================================
# ALTERNATIVE: SIMPLE REWARD-BASED RL (GRPO-like)
# =============================================================================

def train_simple_rl():
    """
    If you want actual RL instead of DPO, here's a minimal example.
    This uses REINFORCE with a simple reward function.

    NOT RECOMMENDED for most cases - DPO is usually better.
    """
    print("=" * 60)
    print("SIMPLE RL TRAINING (REINFORCE)")
    print("=" * 60)
    print("WARNING: DPO is usually better. Use train_dpo.py instead.")

    # This would require:
    # 1. Environment that executes actions and returns rewards
    # 2. Policy gradient (REINFORCE) or PPO
    # 3. Much more compute

    # Pseudo-code:
    """
    for episode in range(num_episodes):
        # Sample trajectory
        state = env.reset()  # Screenshot
        actions = []
        rewards = []

        for step in range(max_steps):
            # Model generates action
            action = model.generate(state)
            actions.append(action)

            # Execute in environment
            next_state, reward, done = env.step(action)
            rewards.append(reward)

            if done:
                break
            state = next_state

        # Compute returns
        returns = compute_returns(rewards)

        # Policy gradient update
        loss = -sum(log_prob(action) * return for action, return in zip(actions, returns))
        loss.backward()
        optimizer.step()
    """

    print("\nFor actual RL, consider using:")
    print("  - TRL's PPOTrainer")
    print("  - OpenAI's Spinning Up")
    print("  - Stable Baselines3")
    print("\nBut seriously, just use DPO. It's simpler and works better.")


if __name__ == "__main__":
    main()
