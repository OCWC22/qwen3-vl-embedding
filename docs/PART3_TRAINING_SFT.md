# PART 3: SUPERVISED FINE-TUNING (SFT)
## Training Generation Models: 3B, A3B (MoE), 32B, A30B (MoE)

---

## 3.1 WHAT IS SFT?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SUPERVISED FINE-TUNING (SFT)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GOAL: Teach the model to produce desired outputs given inputs             │
│                                                                             │
│  BEFORE SFT (Base Model):                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: <image>screenshot</image> Click the login button            │   │
│  │ Output: "I am a large language model trained by..."  (USELESS)      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  AFTER SFT:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: <image>screenshot</image> Click the login button            │   │
│  │ Output: "I'll click the login button for you.                       │   │
│  │         ACTION: CLICK(512, 420)"  (USEFUL)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  TRAINING OBJECTIVE: Next-Token Prediction (Causal Language Modeling)      │
│                                                                             │
│  Loss = -Σ log P(token_t | token_1, ..., token_{t-1}, image)              │
│                                                                             │
│  Only compute loss on ASSISTANT tokens, not USER tokens                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.2 MODEL COMPARISON: DENSE vs MoE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DENSE vs MIXTURE-OF-EXPERTS (MoE)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DENSE MODEL (3B, 8B, 32B, 72B):                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Input ──► [All Layers Active] ──► Output                          │   │
│  │                                                                     │   │
│  │  Every parameter participates in every forward pass                │   │
│  │  FLOPs = O(parameters)                                              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  MoE MODEL (A3B, A30B):                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Input ──► [Router] ──► Select top-K experts ──► Output            │   │
│  │              │                                                      │   │
│  │              ▼                                                      │   │
│  │     ┌──────────────────────────────────────┐                       │   │
│  │     │ Expert 1 │ Expert 2 │ ... │ Expert N │                       │   │
│  │     │  (3B)    │  (3B)    │     │  (3B)    │                       │   │
│  │     └──────────────────────────────────────┘                       │   │
│  │     Only K experts active per token (K=2 typically)                │   │
│  │                                                                     │   │
│  │  Total params: 30B (A3B) or 235B (A30B)                            │   │
│  │  Active params: 3B (A3B) or 30B (A30B)                             │   │
│  │  FLOPs = O(active_parameters) << O(total_parameters)              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  COMPARISON TABLE:                                                          │
│                                                                             │
│  │ Property          │ Dense 32B   │ MoE A3B     │ MoE A30B    │          │
│  ├───────────────────┼─────────────┼─────────────┼─────────────┤          │
│  │ Total Parameters  │ 32B         │ 30B         │ 235B        │          │
│  │ Active Parameters │ 32B         │ 3B          │ 30B         │          │
│  │ Inference Speed   │ 1x          │ 10x faster  │ 1x          │          │
│  │ Memory (FP16)     │ 65GB        │ 60GB        │ 470GB       │          │
│  │ Memory (4-bit)    │ 20GB        │ 20GB        │ 150GB       │          │
│  │ Quality           │ Excellent   │ Very Good   │ Best        │          │
│  │ Training Ease     │ Simple      │ Complex     │ Very Complex│          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MoE-Specific Training Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Expert collapse** | All tokens route to same expert | Auxiliary load balancing loss |
| **Load imbalance** | Some experts underutilized | Capacity factor, expert-choice routing |
| **Gradient noise** | Sparse gradients are noisy | Larger batch sizes |
| **Memory overhead** | Must store all experts | Expert parallelism, offloading |

---

## 3.3 TRAINING REQUIREMENTS BY MODEL

### Dense Models

| Model | Min Data | VRAM (FP16) | VRAM (QLoRA) | Time (8xA100) |
|-------|----------|-------------|--------------|---------------|
| **3B** | 50 conv | 8GB | 4GB | 30 min |
| **8B** | 100 conv | 17GB | 8GB | 1 hour |
| **32B** | 200 conv | 65GB | 20GB | 4 hours |
| **72B** | 500 conv | 145GB | 45GB | 12 hours |

### MoE Models

| Model | Min Data | VRAM (FP16) | VRAM (QLoRA) | Time (8xA100) | Notes |
|-------|----------|-------------|--------------|---------------|-------|
| **A3B** | 100 conv | 60GB | 20GB | 2 hours | Easy to train |
| **A30B** | 300 conv | 470GB | 150GB | 8 hours | Needs 8+ GPUs |

---

## 3.4 SFT TRAINING PIPELINE

### Data Format

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "screenshots/login_screen.png"},
                {"type": "text", "text": "I need to log into my account"}
            ]
        },
        {
            "role": "assistant",
            "content": "I can see the Higgsfield login page. Let me help you log in.\n\nI'll click on the email input field first.\n\nACTION: CLICK(512, 300)"
        }
    ]
}
```

### Training Script

```python
# SFT TRAINING FOR ALL MODEL SIZES

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import torch

# ============================================================================
# MODEL CONFIGURATION BY SIZE
# ============================================================================

MODEL_CONFIGS = {
    "3B": {
        "model_name": "Qwen/Qwen3-VL-3B-Instruct",
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "learning_rate": 2e-5,
        "epochs": 3,
        "min_data": 50,
    },
    "A3B": {  # MoE
        "model_name": "Qwen/Qwen3-VL-A3B-Instruct",
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 1e-5,
        "epochs": 2,
        "min_data": 100,
        # MoE-specific
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate",  # Router
            "experts.0.up_proj", "experts.0.down_proj",  # Expert layers
        ],
    },
    "32B": {
        "model_name": "Qwen/Qwen3-VL-32B-Instruct",
        "lora_r": 64,
        "lora_alpha": 128,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "learning_rate": 1e-5,
        "epochs": 2,
        "min_data": 200,
    },
    "A30B": {  # MoE
        "model_name": "Qwen/Qwen3-VL-A30B-Instruct",
        "lora_r": 64,
        "lora_alpha": 128,
        "batch_size": 1,
        "gradient_accumulation": 32,
        "learning_rate": 5e-6,
        "epochs": 2,
        "min_data": 300,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate",
            # Note: A30B has many experts, LoRA subset recommended
        ],
    },
}

def train_sft(model_size: str, data_path: str, output_dir: str):
    """Train SFT for any model size."""

    config = MODEL_CONFIGS[model_size]

    # Load model with quantization for memory efficiency
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        # For QLoRA
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    processor = AutoProcessor.from_pretrained(config["model_name"])

    # LoRA configuration
    target_modules = config.get("lora_target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Should be ~1-2%

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # For MoE models
        ddp_find_unused_parameters=False if "A" in model_size else None,
    )

    # Custom data collator for vision-language
    def collate_fn(examples):
        texts = []
        images = []

        for example in examples:
            messages = example["messages"]
            # Format as chat
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            # Extract images
            for msg in messages:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content.get("type") == "image":
                            images.append(content["image"])

        # Process batch
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Create labels (mask user tokens)
        batch["labels"] = batch["input_ids"].clone()
        # Mask padding
        batch["labels"][batch["attention_mask"] == 0] = -100

        return batch

    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    # Train
    trainer.train()

    # Save LoRA weights
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    return model, processor


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=["3B", "A3B", "32B", "A30B"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    train_sft(args.model_size, args.data_path, args.output_dir)
```

### Multi-GPU Training Commands

```bash
# Single GPU - 3B model
python train_sft.py \
    --model_size 3B \
    --data_path data/sft_train.jsonl \
    --output_dir checkpoints/sft-3b-qlora

# Multi-GPU - A3B MoE (needs ~80GB total)
torchrun --nproc_per_node=2 train_sft.py \
    --model_size A3B \
    --data_path data/sft_train.jsonl \
    --output_dir checkpoints/sft-a3b-qlora

# Multi-GPU - 32B Dense (needs ~160GB total)
torchrun --nproc_per_node=4 train_sft.py \
    --model_size 32B \
    --data_path data/sft_train.jsonl \
    --output_dir checkpoints/sft-32b-qlora

# Multi-Node - A30B MoE (needs ~600GB total)
# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 \
    train_sft.py --model_size A30B ...

# Node 2:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 \
    train_sft.py --model_size A30B ...
```

---

## 3.5 SFT SHORTCOMINGS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SFT LIMITATIONS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. EXPOSURE BIAS                                                           │
│     ─────────────                                                           │
│     Problem: Model trains on ground truth, but at inference uses its own   │
│              predictions. If it makes one mistake, errors compound.        │
│                                                                             │
│     Train: "Click" → "the" → "button" (all correct tokens)                 │
│     Infer: "Click" → "a" → ??? (never saw what comes after "a")           │
│                                                                             │
│     Mitigation: Scheduled sampling, RL fine-tuning                          │
│                                                                             │
│  2. DISTRIBUTION MISMATCH                                                   │
│     ──────────────────────                                                  │
│     Problem: Training data may not cover all situations                    │
│                                                                             │
│     If training data has no error states → model fails on errors          │
│     If training data is all English → model fails on other languages      │
│                                                                             │
│     Mitigation: Diverse data augmentation, RL exploration                   │
│                                                                             │
│  3. MODE COLLAPSE                                                           │
│     ─────────────                                                           │
│     Problem: Model learns to produce single "average" response             │
│                                                                             │
│     If multiple valid actions exist, model may blend them into nonsense   │
│                                                                             │
│     Mitigation: Temperature sampling, diverse examples                      │
│                                                                             │
│  4. NO PREFERENCE LEARNING                                                  │
│     ──────────────────────                                                  │
│     Problem: SFT only learns "what to do", not "what's better"            │
│                                                                             │
│     Both responses correct, but one is better:                             │
│     A: "Click the login button" (good)                                     │
│     B: "I'll help you login. Let me click the button." (better)           │
│                                                                             │
│     SFT treats them the same. Need DPO/RLHF for preferences.              │
│                                                                             │
│  5. CATASTROPHIC FORGETTING                                                 │
│     ────────────────────────                                                │
│     Problem: Fine-tuning on task data hurts general abilities             │
│                                                                             │
│     After SFT on Higgsfield, model may forget general VQA                  │
│                                                                             │
│     Mitigation: LoRA (preserves base weights), low LR, short training     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.6 SFT EVALUATION

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Perplexity** | How surprised model is by correct outputs | Lower is better, <5 for good SFT |
| **Action Accuracy** | % of correctly generated actions | >85% |
| **Format Compliance** | % of outputs with correct ACTION format | >95% |
| **Task Success Rate** | % of tasks completed correctly | >70% |
| **BLEU/ROUGE** | N-gram overlap with reference | Not critical for actions |

### Evaluation Script

```python
def evaluate_sft(model, processor, test_data):
    """Evaluate SFT model on test set."""

    results = {
        "action_accuracy": 0,
        "format_compliance": 0,
        "perplexity": 0,
        "total": 0,
    }

    for example in test_data:
        # Generate
        inputs = processor(
            text=example["prompt"],
            images=example["images"],
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy for eval
            )

        generated = processor.decode(outputs[0], skip_special_tokens=True)

        # Check format
        if "ACTION:" in generated:
            results["format_compliance"] += 1

        # Check action correctness
        predicted_action = extract_action(generated)
        correct_action = extract_action(example["reference"])

        if actions_match(predicted_action, correct_action):
            results["action_accuracy"] += 1

        results["total"] += 1

    # Compute percentages
    for key in ["action_accuracy", "format_compliance"]:
        results[key] = results[key] / results["total"] * 100

    return results
```

---

## 3.7 WHICH MODEL SIZE TO CHOOSE?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL SIZE DECISION TREE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  START: What's your constraint?                                             │
│                                                                             │
│  ┌─ SINGLE GPU (24-48GB)?                                                  │
│  │   │                                                                      │
│  │   ├─ Speed critical? ──────────────► 3B Dense                           │
│  │   │                                                                      │
│  │   └─ Quality > Speed? ─────────────► A3B MoE (best bang for buck)       │
│  │                                                                          │
│  ├─ MULTI-GPU (2-4 × 80GB)?                                                │
│  │   │                                                                      │
│  │   ├─ Simple training preferred? ───► 32B Dense                          │
│  │   │                                                                      │
│  │   └─ Maximum quality? ─────────────► A3B MoE + larger batch             │
│  │                                                                          │
│  └─ GPU CLUSTER (8+ × 80GB)?                                               │
│      │                                                                      │
│      ├─ Inference cost matters? ──────► 32B Dense or A3B MoE               │
│      │                                                                      │
│      └─ Maximum quality, cost no issue? ► A30B MoE                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  FOR HIGGSFIELD AUTOMATION:                                                 │
│                                                                             │
│  RECOMMENDED: Qwen3-VL-A3B (MoE)                                           │
│  - 30B total params, 3B active → quality of 30B, speed of 3B              │
│  - Fits on 1× A100-80GB with QLoRA                                         │
│  - Great quality/cost tradeoff                                              │
│                                                                             │
│  ALTERNATIVE: Qwen3-VL-3B (Dense)                                          │
│  - If you only have 24GB GPU                                               │
│  - Faster training and inference                                            │
│  - ~10% lower quality than A3B                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## NEXT: Part 4 - Offline RL (DPO)

Continue to [PART4_OFFLINE_RL_DPO.md](./PART4_OFFLINE_RL_DPO.md)
