# PART 2: TRAINING PIPELINE - EMBEDDING & RERANKER
## Complete Training Guide with Shortcomings and Solutions

---

## 2.1 EMBEDDING MODEL TRAINING

### Training Objective

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTRASTIVE LEARNING (InfoNCE)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GOAL: Make similar things close, dissimilar things far apart              │
│                                                                             │
│  POSITIVE PAIR:                                                             │
│  ┌─────────────┐                        ┌─────────────┐                    │
│  │ Login       │   should be close to   │ "Click the  │                    │
│  │ Screenshot  │ ◄────────────────────► │  email      │                    │
│  │             │   in embedding space   │  field"     │                    │
│  └─────────────┘                        └─────────────┘                    │
│                                                                             │
│  NEGATIVE PAIRS:                                                            │
│  ┌─────────────┐                        ┌─────────────┐                    │
│  │ Login       │   should be FAR from   │ "Click      │                    │
│  │ Screenshot  │ ◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ► │  Settings"  │                    │
│  │             │                        │             │                    │
│  └─────────────┘                        └─────────────┘                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  LOSS FUNCTION: InfoNCE (Noise Contrastive Estimation)                      │
│                                                                             │
│                         exp(sim(q, d+) / τ)                                 │
│  L = -log ──────────────────────────────────────────                        │
│           exp(sim(q, d+) / τ) + Σ exp(sim(q, d-) / τ)                      │
│                                                                             │
│  Where:                                                                     │
│  - q = query embedding                                                      │
│  - d+ = positive document embedding                                         │
│  - d- = negative document embeddings                                        │
│  - τ = temperature (typically 0.02-0.05)                                   │
│  - sim = cosine similarity                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

```python
# EMBEDDING TRAINING PIPELINE

# Step 1: Data Preparation
"""
Data format (JSONL):
{
    "query": {"text": "...", "image": "path/to/screenshot.png"},
    "positive": {"text": "...", "image": "path/to/screenshot.png"},
    "negative": {"text": "...", "image": "path/to/screenshot.png"}
}
"""

# Step 2: Model Setup
from transformers import Qwen2_5_VLForConditionalGeneration
import torch.nn as nn

class Qwen3VLEmbedding(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.temperature = 0.02

    def get_embedding(self, input_ids, attention_mask, pixel_values):
        """Extract embedding from EOS token position."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        # Get last hidden state
        hidden = outputs.hidden_states[-1]
        # Pool at EOS token (last non-padding token)
        eos_positions = attention_mask.sum(dim=1) - 1
        embeddings = hidden[range(len(hidden)), eos_positions]
        # L2 normalize
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(self, query, positive, negatives):
        """Compute contrastive loss."""
        q_emb = self.get_embedding(**query)      # [B, D]
        p_emb = self.get_embedding(**positive)   # [B, D]
        n_emb = self.get_embedding(**negatives)  # [B*N, D]

        # In-batch negatives + hard negatives
        # Similarity matrix
        pos_sim = (q_emb * p_emb).sum(-1) / self.temperature  # [B]
        neg_sim = torch.mm(q_emb, n_emb.T) / self.temperature  # [B, B*N]

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss

# Step 3: Training Configuration
training_config = {
    "learning_rate": 1e-5,
    "batch_size": 4,  # Per GPU
    "gradient_accumulation": 8,
    "effective_batch_size": 32,  # Important for contrastive learning
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 2048,
    "temperature": 0.02,
}

# Step 4: QLoRA Setup (for memory efficiency)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Hard Negative Mining (CRITICAL)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARD NEGATIVE MINING STRATEGIES                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHY HARD NEGATIVES MATTER:                                                 │
│                                                                             │
│  EASY NEGATIVES:                                                            │
│  Query: "Login to the website"                                              │
│  Negative: "Read documentation about Python"                                │
│  → Model learns nothing useful, too obviously different                     │
│                                                                             │
│  HARD NEGATIVES:                                                            │
│  Query: "Login to the website"                                              │
│  Negative: "Sign up for a new account"                                      │
│  → Model must learn subtle differences, much more useful                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  MINING STRATEGIES:                                                         │
│                                                                             │
│  1. BM25 MINING                                                             │
│     - Use keyword matching to find similar but wrong documents              │
│     - Fast, good baseline                                                   │
│     - Code: elasticsearch/pyserini                                          │
│                                                                             │
│  2. EMBEDDING MINING                                                        │
│     - Use current model to find near-miss negatives                        │
│     - Find documents with similarity 0.6-0.9 (similar but not identical)   │
│     - Requires periodic re-mining as model improves                         │
│                                                                             │
│  3. IN-BATCH NEGATIVES                                                      │
│     - Use other queries' positives as negatives                            │
│     - Free, no extra computation                                            │
│     - Larger batch = more negatives = better                                │
│                                                                             │
│  4. CROSS-BATCH NEGATIVES (MoCo-style)                                      │
│     - Maintain queue of recent embeddings                                   │
│     - Use as additional negatives                                           │
│     - Memory efficient, increases effective negatives                       │
│                                                                             │
│  5. LLM-GENERATED NEGATIVES                                                 │
│     - Use GPT-4 to generate plausible but wrong actions                    │
│     - Highest quality, expensive                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Shortcomings of Embedding Training

| Shortcoming | Impact | Mitigation |
|-------------|--------|------------|
| **Batch size sensitivity** | Small batches = poor performance | Gradient accumulation, distributed training |
| **Hard negative quality** | Bad negatives = wasted training | Multi-strategy mining, LLM generation |
| **Dimension bottleneck** | 1024-dim loses information | Higher dimensions, multiple embeddings |
| **Modality gap** | Image vs text embeddings don't align | Shared projection, contrastive pretraining |
| **Temperature tuning** | Wrong τ = poor clustering | Grid search, adaptive temperature |
| **Catastrophic forgetting** | Loses general knowledge | LoRA, low learning rate |

---

## 2.2 RERANKER MODEL TRAINING

### Training Objective

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY CLASSIFICATION (BCE Loss)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GOAL: Learn to score relevance of (query, document) pairs                 │
│                                                                             │
│  INPUT:                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ <image>login_screen.png</image>                                      │  │
│  │ Query: I need to enter my email to login                             │  │
│  │ Document: Click the email input field in the center of the form     │  │
│  │ Is this document relevant? Answer yes or no.                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  OUTPUT: "yes" or "no"                                                      │
│                                                                             │
│  SCORING:                                                                   │
│  score = logit("yes") - logit("no")                                        │
│                                                                             │
│  LOSS: Binary Cross-Entropy                                                 │
│  L = -[y·log(σ(score)) + (1-y)·log(1-σ(score))]                           │
│                                                                             │
│  Where:                                                                     │
│  - y = 1 for relevant (positive) pairs                                     │
│  - y = 0 for irrelevant (negative) pairs                                   │
│  - σ = sigmoid function                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

```python
# RERANKER TRAINING PIPELINE

# Step 1: Data Preparation
"""
Data format (JSONL):
{
    "query": {"text": "...", "image": "path/to/screenshot.png"},
    "document": "action description",
    "label": 1  # or 0
}

CRITICAL: Balance your labels!
- 50% positive (label=1)
- 50% negative (label=0)
"""

# Step 2: Model Setup
class Qwen3VLReranker(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get token IDs for "yes" and "no"
        self.yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

    def forward(self, input_ids, attention_mask, pixel_values, labels):
        """Compute reranking score and loss."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        # Get logits for last token (where model predicts yes/no)
        last_logits = outputs.logits[:, -1, :]

        # Extract yes/no logits
        yes_logits = last_logits[:, self.yes_token_id]
        no_logits = last_logits[:, self.no_token_id]

        # Score = difference
        scores = yes_logits - no_logits

        # BCE loss
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())

        return {"loss": loss, "scores": scores}

# Step 3: Prompt Template
RERANKER_PROMPT = """<image>{image}</image>

Query: {query_text}

Document: {document}

Is this document relevant to the query and image? Answer with only "yes" or "no"."""

# Step 4: Training Configuration
training_config = {
    "learning_rate": 2e-5,
    "batch_size": 8,
    "gradient_accumulation": 4,
    "epochs": 2,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 1024,
}

# Step 5: QLoRA (same as embedding)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)
```

### Listwise vs Pointwise Training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POINTWISE vs PAIRWISE vs LISTWISE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POINTWISE (what we described above):                                       │
│  - Score each (query, doc) pair independently                              │
│  - Label: relevant (1) or not (0)                                          │
│  - Simple, but ignores relative ranking                                     │
│                                                                             │
│  PAIRWISE (RankNet, LambdaRank):                                           │
│  - Given (query, doc_a, doc_b), predict which is more relevant            │
│  - Learns relative preferences                                              │
│  - Better ranking, more complex data                                        │
│                                                                             │
│  LISTWISE (ListMLE, LambdaMART):                                           │
│  - Given (query, [doc_1, doc_2, ..., doc_n]), optimize full ranking       │
│  - Best theoretical ranking                                                 │
│  - Most complex, needs ranked lists                                        │
│                                                                             │
│  RECOMMENDATION FOR YOUR CASE:                                              │
│  Start with POINTWISE (simpler), upgrade to PAIRWISE if needed             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Shortcomings of Reranker Training

| Shortcoming | Impact | Mitigation |
|-------------|--------|------------|
| **Label imbalance** | Model predicts all 0 or all 1 | Enforce 50/50 balance |
| **Position bias** | Model learns to rank by position | Shuffle document order |
| **Length bias** | Prefers longer/shorter documents | Normalize by length |
| **Listwise gap** | Pointwise ≠ good ranking | Use pairwise/listwise losses |
| **Slow inference** | Must score each pair | Batch efficiently, limit candidates |
| **Instruction sensitivity** | Different prompts = different scores | Standardize prompts |

---

## 2.3 TRAINING REQUIREMENTS BY MODEL SIZE

### Embedding Model

| Model | Min Data | VRAM (QLoRA) | Time (1xA100) | Effective Batch |
|-------|----------|--------------|---------------|-----------------|
| **2B** | 250 triplets | 8GB | 1 hour | 128 |
| **8B** | 500 triplets | 24GB | 3 hours | 64 |

### Reranker Model

| Model | Min Data | VRAM (QLoRA) | Time (1xA100) | Batch Size |
|-------|----------|--------------|---------------|------------|
| **2B** | 150 pairs | 8GB | 30 min | 32 |
| **8B** | 300 pairs | 24GB | 1.5 hours | 16 |

### Training Commands

```bash
# Embedding training (8B model, QLoRA)
python train_embedding.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --data_path data/embedding_train.jsonl \
    --output_dir checkpoints/embedding-8b-qlora \
    --lora_r 32 \
    --lora_alpha 64 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 1e-5 \
    --epochs 3 \
    --temperature 0.02 \
    --bf16

# Reranker training (8B model, QLoRA)
python train_reranker.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --data_path data/reranker_train.jsonl \
    --output_dir checkpoints/reranker-8b-qlora \
    --lora_r 16 \
    --lora_alpha 32 \
    --batch_size 8 \
    --gradient_accumulation 4 \
    --learning_rate 2e-5 \
    --epochs 2 \
    --bf16
```

---

## 2.4 EVALUATION METRICS

### Embedding Evaluation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING EVALUATION METRICS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RECALL@K                                                                   │
│  - Is the correct document in top-K retrieved?                             │
│  - Recall@1, Recall@5, Recall@10, Recall@100                               │
│  - Higher is better                                                         │
│                                                                             │
│  Example:                                                                   │
│  Query: "Login to website"                                                  │
│  Top 5 retrieved: [doc_3, doc_7, doc_1, doc_9, doc_2]                      │
│  Correct: doc_1                                                             │
│  → Recall@1 = 0 (doc_1 not in top 1)                                       │
│  → Recall@5 = 1 (doc_1 is in top 5)                                        │
│                                                                             │
│  MRR (Mean Reciprocal Rank)                                                 │
│  - Average of 1/rank of first correct document                             │
│  - MRR = 1/3 for the example above (correct at position 3)                 │
│                                                                             │
│  NDCG@K (Normalized Discounted Cumulative Gain)                            │
│  - Considers graded relevance and position                                  │
│  - Better for multiple relevant documents                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Reranker Evaluation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RERANKER EVALUATION METRICS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ACCURACY                                                                   │
│  - % of (query, doc) pairs correctly classified                            │
│  - Threshold at score = 0                                                   │
│                                                                             │
│  RERANKING METRICS (after reranking embedding results):                    │
│  - MRR improvement over embedding baseline                                  │
│  - NDCG@5 improvement                                                       │
│  - Recall@1 improvement                                                     │
│                                                                             │
│  Example:                                                                   │
│  Embedding Recall@1: 45%                                                    │
│  After Reranking Recall@1: 72%                                             │
│  → +27% improvement (good!)                                                 │
│                                                                             │
│  AUC-ROC                                                                    │
│  - Area under receiver operating curve                                      │
│  - Measures discrimination ability                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Target Metrics for Higgsfield

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Embedding Recall@10 | 70% | 85% | 95% |
| Embedding Recall@100 | 90% | 95% | 99% |
| Embedding MRR | 0.4 | 0.6 | 0.8 |
| Reranker Accuracy | 75% | 85% | 92% |
| Reranker NDCG@5 | 0.7 | 0.85 | 0.95 |
| End-to-end Recall@1 | 60% | 75% | 85% |

---

## 2.5 COMMON TRAINING FAILURES

### Embedding Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss doesn't decrease | LR too low/high | Grid search LR 1e-6 to 1e-4 |
| Loss decreases but metrics don't | Bad negatives | Improve hard negative mining |
| All embeddings collapse to same point | Temperature too high | Lower τ to 0.01-0.02 |
| Overfitting (train >> val) | Too few negatives | Increase batch size, add negatives |
| Underfitting | Model too small | Use 8B instead of 2B |

### Reranker Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Predicts all 1s | Label imbalance | Ensure 50/50 positive/negative |
| Predicts all 0s | Label imbalance | Ensure 50/50 positive/negative |
| Accuracy ~50% | Not learning | Check data quality, increase LR |
| Train acc high, val acc low | Overfitting | Add dropout, reduce epochs |
| Scores all similar | Prompt issue | Check template, add examples |

---

## NEXT: Part 3 - SFT Training for All Model Sizes

Continue to [PART3_TRAINING_SFT.md](./PART3_TRAINING_SFT.md)
