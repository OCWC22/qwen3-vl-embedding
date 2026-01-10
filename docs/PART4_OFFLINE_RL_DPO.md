# PART 4: OFFLINE REINFORCEMENT LEARNING (DPO)
## Direct Preference Optimization - Learning What's Better

---

## 4.1 WHAT IS DPO?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIRECT PREFERENCE OPTIMIZATION (DPO)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SFT teaches: "This is a correct response"                                 │
│  DPO teaches: "This response is BETTER than that response"                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PROMPT:                                                             │   │
│  │ <image>login_screen.png</image>                                     │   │
│  │ Log me in                                                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ CHOSEN (better):                                                    │   │
│  │ "I'll click the email field now.                                    │   │
│  │  ACTION: CLICK(512, 300)"                                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ REJECTED (worse):                                                   │   │
│  │ "I can help you log in. First, you'll need to enter your email.   │   │
│  │  Would you like me to explain the process? There are several       │   │
│  │  options available for authentication..."                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Both responses are "correct" in some sense, but CHOSEN is:               │
│  ✓ More direct                                                             │
│  ✓ Has ACTION command                                                      │
│  ✓ Less verbose                                                            │
│  ✓ Actually helpful                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 WHY DPO INSTEAD OF PPO?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DPO vs PPO COMPARISON                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PPO (Proximal Policy Optimization) - Online RL                            │
│  ───────────────────────────────────────────────                           │
│                                                                             │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐            │
│  │  Policy   │──►│  Generate │──►│  Reward   │──►│   Value   │            │
│  │  Model    │   │ Response  │   │   Model   │   │   Model   │            │
│  │   (LLM)   │   │           │   │  (Critic) │   │  (Critic) │            │
│  └───────────┘   └───────────┘   └───────────┘   └───────────┘            │
│       32B            ↑               32B             32B                    │
│                      │                                                      │
│                 Reference                                                   │
│                  Model                                                      │
│                   32B                                                       │
│                                                                             │
│  TOTAL MEMORY: 4 × 32B = 128B parameters loaded!                           │
│  COMPLEXITY: High (reward model, value model, reference model)             │
│  STABILITY: Low (reward hacking, mode collapse)                            │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  DPO (Direct Preference Optimization) - Offline RL                         │
│  ────────────────────────────────────────────────                          │
│                                                                             │
│  ┌───────────┐                       ┌───────────┐                         │
│  │  Policy   │ ◄─── Compare ────────►│ Reference │                         │
│  │  Model    │      probabilities    │   Model   │                         │
│  │   (LLM)   │                       │   (LLM)   │                         │
│  └───────────┘                       └───────────┘                         │
│       32B                                 32B                               │
│                                                                             │
│  TOTAL MEMORY: 2 × 32B = 64B parameters (or just 1 with tricks)           │
│  COMPLEXITY: Low (just supervised learning on pairs)                       │
│  STABILITY: High (no reward model to hack)                                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COMPARISON TABLE:                                                          │
│                                                                             │
│  │ Aspect           │ PPO              │ DPO               │              │
│  ├──────────────────┼──────────────────┼───────────────────┤              │
│  │ Memory           │ 4× model size    │ 2× model size     │              │
│  │ Training Time    │ Days/Weeks       │ Hours             │              │
│  │ Stability        │ Low              │ High              │              │
│  │ Data Required    │ Reward model +   │ Preference pairs  │              │
│  │                  │ environment      │ only              │              │
│  │ Quality Ceiling  │ Higher           │ Good enough       │              │
│  │ Reward Hacking   │ Common problem   │ Not possible      │              │
│  │ Implementation   │ Complex          │ Simple            │              │
│                                                                             │
│  VERDICT: Use DPO first. Only use PPO if DPO plateaus.                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.3 DPO MATHEMATICAL FORMULATION

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DPO LOSS FUNCTION                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KEY INSIGHT: DPO derives the optimal policy directly from preferences,    │
│  without explicitly learning a reward model.                                │
│                                                                             │
│  Standard RLHF Objective:                                                   │
│                                                                             │
│  maximize  E[r(x,y)] - β·KL[π_θ(y|x) || π_ref(y|x)]                        │
│     π_θ                                                                     │
│                                                                             │
│  Where:                                                                     │
│  - π_θ = policy (model being trained)                                      │
│  - π_ref = reference model (frozen SFT model)                              │
│  - r(x,y) = reward for response y given prompt x                           │
│  - β = KL penalty coefficient                                               │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  DPO LOSS:                                                                  │
│                                                                             │
│  L_DPO(π_θ; π_ref) = -E[ log σ( β · (                                      │
│      log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))          │
│  ))]                                                                        │
│                                                                             │
│  Simplified:                                                                │
│  L = -log σ( β · (log_ratio_chosen - log_ratio_rejected) )                 │
│                                                                             │
│  Where:                                                                     │
│  - y_w = chosen (winner) response                                          │
│  - y_l = rejected (loser) response                                         │
│  - σ = sigmoid function                                                     │
│  - β = typically 0.1 to 0.5                                                │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  INTUITION:                                                                 │
│  - Increase probability of chosen response relative to reference           │
│  - Decrease probability of rejected response relative to reference         │
│  - KL term prevents model from drifting too far from reference             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.4 DPO TRAINING PIPELINE

### Data Format

```json
{
    "prompt": [
        {"type": "image", "image": "screenshots/generate_screen.png"},
        {"type": "text", "text": "Generate a video of a dancing cat"}
    ],
    "chosen": "I'll start the generation now.\n\nACTION: CLICK(700, 500)\n\nThis clicks the Generate button. The video will be ready in about 30 seconds.",
    "rejected": "I'd be happy to help you generate a video! Before we start, you should know there are several options. You could adjust the duration, change the resolution, or modify the style. Would you like me to explain each option? Also, make sure you have enough credits in your account."
}
```

### Training Script

```python
# DPO TRAINING PIPELINE

from transformers import AutoModelForCausalLM, AutoProcessor
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# ============================================================================
# DPO CONFIGURATION
# ============================================================================

DPO_CONFIGS = {
    "3B": {
        "model_name": "Qwen/Qwen3-VL-3B-Instruct",
        "sft_checkpoint": "checkpoints/sft-3b-qlora",  # Start from SFT
        "beta": 0.1,  # KL penalty
        "learning_rate": 5e-6,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "epochs": 1,
        "min_data": 25,
    },
    "A3B": {
        "model_name": "Qwen/Qwen3-VL-A3B-Instruct",
        "sft_checkpoint": "checkpoints/sft-a3b-qlora",
        "beta": 0.1,
        "learning_rate": 2e-6,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "epochs": 1,
        "min_data": 50,
    },
    "32B": {
        "model_name": "Qwen/Qwen3-VL-32B-Instruct",
        "sft_checkpoint": "checkpoints/sft-32b-qlora",
        "beta": 0.05,  # Lower beta for larger models
        "learning_rate": 1e-6,
        "batch_size": 1,
        "gradient_accumulation": 32,
        "epochs": 1,
        "min_data": 100,
    },
}

def train_dpo(model_size: str, data_path: str, output_dir: str):
    """Train DPO for any model size."""

    config = DPO_CONFIGS[model_size]

    # Load SFT model as starting point
    model = AutoModelForCausalLM.from_pretrained(
        config["sft_checkpoint"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # Reference model (frozen copy of SFT model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config["sft_checkpoint"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(config["model_name"])

    # Optional: Add LoRA for memory efficiency
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=config["beta"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        num_train_epochs=config["epochs"],
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        # DPO specific
        loss_type="sigmoid",  # or "hinge", "ipo"
        max_length=2048,
        max_prompt_length=1024,
    )

    # Load dataset
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Format dataset for DPO
    def format_for_dpo(example):
        # Format prompt
        prompt_parts = []
        for item in example["prompt"]:
            if item["type"] == "image":
                prompt_parts.append(f"<image>{item['image']}</image>")
            else:
                prompt_parts.append(item["text"])
        prompt = "\n".join(prompt_parts)

        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    dataset = dataset.map(format_for_dpo)

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(output_dir)

    return model


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    train_dpo("A3B", "data/dpo_train.jsonl", "checkpoints/dpo-a3b")
```

---

## 4.5 CREATING PREFERENCE DATA

### Strategy 1: Human Annotation

```
PROCESS:
1. Generate 2+ responses per prompt
2. Human ranks responses by quality
3. Create (chosen, rejected) pairs

PROS: High quality
CONS: Expensive, slow
```

### Strategy 2: LLM-as-Judge

```python
# Use GPT-4 or Claude to judge which response is better

JUDGE_PROMPT = """
You are evaluating two AI responses for a computer automation task.

Task: {task}
Screenshot: {screenshot_description}

Response A:
{response_a}

Response B:
{response_b}

Which response is better for completing the task? Consider:
1. Directness - Does it immediately take action?
2. Correctness - Is the action correct?
3. Helpfulness - Does it actually help complete the task?
4. Brevity - Is it concise without unnecessary explanation?

Answer with ONLY "A" or "B".
"""

def create_preference_pair(prompt, response_a, response_b):
    """Use LLM judge to create preference pair."""
    judgment = llm_judge(JUDGE_PROMPT.format(...))

    if judgment == "A":
        return {"prompt": prompt, "chosen": response_a, "rejected": response_b}
    else:
        return {"prompt": prompt, "chosen": response_b, "rejected": response_a}
```

### Strategy 3: Rule-Based

```python
# Automatic preference based on rules

def score_response(response: str) -> float:
    """Score a response based on rules."""
    score = 0

    # Has ACTION command = good
    if "ACTION:" in response:
        score += 50

    # Short and direct = good
    if len(response) < 200:
        score += 20
    elif len(response) > 500:
        score -= 20

    # Asks questions = bad
    if "?" in response:
        score -= 30

    # Hedging language = bad
    if any(phrase in response.lower() for phrase in
           ["i think", "maybe", "perhaps", "would you like"]):
        score -= 20

    return score

def create_preference_pairs(prompt, responses):
    """Create pairs from multiple responses."""
    scored = [(r, score_response(r)) for r in responses]
    scored.sort(key=lambda x: x[1], reverse=True)

    pairs = []
    for i in range(len(scored) - 1):
        if scored[i][1] > scored[i+1][1]:  # Must be strictly better
            pairs.append({
                "prompt": prompt,
                "chosen": scored[i][0],
                "rejected": scored[i+1][0],
            })

    return pairs
```

---

## 4.6 DPO SHORTCOMINGS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DPO LIMITATIONS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PREFERENCE DATA QUALITY                                                 │
│     ───────────────────────                                                 │
│     Problem: DPO is only as good as the preference annotations            │
│                                                                             │
│     If annotators disagree or make mistakes → noisy training              │
│     If chosen/rejected are too similar → model learns nothing             │
│     If chosen/rejected are too different → trivial learning               │
│                                                                             │
│     Mitigation: Multiple annotators, clear guidelines, quality control    │
│                                                                             │
│  2. REFERENCE MODEL DRIFT                                                   │
│     ──────────────────────                                                  │
│     Problem: As training progresses, model drifts from reference          │
│                                                                             │
│     This can cause:                                                         │
│     - Reward hacking (finding loopholes in preference)                     │
│     - Mode collapse (all responses become similar)                         │
│     - Forgetting (loses general capabilities)                              │
│                                                                             │
│     Mitigation: Low β, early stopping, eval on diverse tasks              │
│                                                                             │
│  3. OFFLINE NATURE                                                          │
│     ──────────────                                                          │
│     Problem: DPO can only learn from static preference data               │
│                                                                             │
│     Can't explore new behaviors beyond the preference dataset             │
│     Can't adapt to changing environments                                   │
│                                                                             │
│     Mitigation: Iterative DPO (generate → annotate → train → repeat)     │
│                                                                             │
│  4. BINARY PREFERENCES                                                      │
│     ──────────────────                                                      │
│     Problem: DPO only handles "A > B", not degrees of preference          │
│                                                                             │
│     "A is slightly better than B" treated same as                         │
│     "A is vastly better than B"                                           │
│                                                                             │
│     Mitigation: Use margin-based losses, or IPO variant                   │
│                                                                             │
│  5. LENGTH BIAS                                                             │
│     ───────────                                                             │
│     Problem: DPO can learn spurious correlations with length              │
│                                                                             │
│     If chosen responses happen to be shorter → model learns to be terse   │
│     Even when longer response would be better                              │
│                                                                             │
│     Mitigation: Length-normalized loss, balanced preference data          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.7 DPO VARIANTS

| Variant | Description | When to Use |
|---------|-------------|-------------|
| **Standard DPO** | Original sigmoid loss | Default choice |
| **IPO** | Identity preference optimization, margin-based | When preferences have margins |
| **cDPO** | Conservative DPO, adds label smoothing | When data is noisy |
| **rDPO** | Robust DPO, handles label noise | When annotations are unreliable |
| **ORPO** | Odds ratio preference optimization | No reference model needed |
| **SimPO** | Simple preference optimization | Lower memory, competitive quality |

### SimPO (Recommended for Vision-Language)

```python
# SimPO: No reference model needed!

class SimPOLoss:
    """Simple Preference Optimization - no reference model."""

    def __init__(self, beta=2.0, gamma=0.5):
        self.beta = beta
        self.gamma = gamma

    def forward(self, chosen_logprobs, rejected_logprobs,
                chosen_length, rejected_length):
        """
        Compute SimPO loss.

        Uses length-normalized log probabilities and adds
        a margin gamma to prevent reward hacking.
        """
        # Length normalization
        chosen_reward = chosen_logprobs / chosen_length
        rejected_reward = rejected_logprobs / rejected_length

        # Loss with margin
        loss = -F.logsigmoid(
            self.beta * (chosen_reward - rejected_reward) - self.gamma
        )

        return loss.mean()
```

---

## 4.8 DPO TRAINING REQUIREMENTS

| Model | Min Data | VRAM (with ref) | VRAM (SimPO) | Time (1xA100) |
|-------|----------|-----------------|--------------|---------------|
| **3B** | 25 pairs | 16GB | 8GB | 20 min |
| **A3B** | 50 pairs | 120GB | 60GB | 1 hour |
| **32B** | 100 pairs | 130GB | 65GB | 3 hours |
| **A30B** | 200 pairs | 940GB | 470GB | 8 hours |

---

## 4.9 WHEN TO USE DPO vs SKIP IT

```
USE DPO WHEN:
✓ SFT model is verbose and hedges
✓ SFT model asks unnecessary questions
✓ SFT model doesn't output ACTION consistently
✓ You have preference data (or can create it)
✓ Quality improvement is worth 2-3 hours training

SKIP DPO WHEN:
✗ SFT model is already direct and action-oriented
✗ No preference data available
✗ Time/resource constrained
✗ Will do online RL (PPO/GRPO) anyway

FOR HIGGSFIELD:
→ DO IT. DPO will make the agent more direct and action-oriented.
→ Expected improvement: +5-10% task success rate
```

---

## NEXT: Part 5 - Online RL (PPO/GRPO)

Continue to [PART5_ONLINE_RL.md](./PART5_ONLINE_RL.md)
