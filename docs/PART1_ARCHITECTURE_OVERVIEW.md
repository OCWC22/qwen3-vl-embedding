# PART 1: ARCHITECTURE OVERVIEW
## Complete Guide to Qwen3-VL Model Family for Computer Use Agents

---

## 1.1 THE COMPLETE MODEL STACK

You're building a computer use agent. Here's every model you need and what each does:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE MODEL STACK                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   EMBEDDING     │     │    RERANKER     │     │   GENERATION    │       │
│  │     MODEL       │────▶│     MODEL       │────▶│     MODEL       │       │
│  │  (Retrieval)    │     │   (Ranking)     │     │   (Action)      │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│         │                        │                       │                  │
│         │                        │                       │                  │
│         ▼                        ▼                       ▼                  │
│  "Find 100 similar        "Pick best 5           "Execute the              │
│   past situations"         from 100"              best action"             │
│                                                                             │
│  Speed: 10ms              Speed: 500ms           Speed: 200-2000ms         │
│  Purpose: Recall          Purpose: Precision     Purpose: Execution        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                        TRAINING METHODS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │      SFT        │     │   OFFLINE RL    │     │   ONLINE RL     │       │
│  │  (Supervised)   │────▶│     (DPO)       │────▶│  (PPO/GRPO)     │       │
│  │                 │     │                 │     │                 │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│         │                        │                       │                  │
│         ▼                        ▼                       ▼                  │
│  "Learn from             "Learn what's            "Learn from              │
│   demonstrations"         better vs worse"         trial and error"        │
│                                                                             │
│  Data: 100+ examples     Data: 50+ pairs          Data: Environment        │
│  Cost: $                 Cost: $$                 Cost: $$$$               │
│  Stability: High         Stability: Medium        Stability: Low           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.2 MODEL SIZES AVAILABLE

### Qwen3-VL Model Family

| Model | Parameters | Type | VRAM (FP16) | VRAM (4-bit) | Use Case |
|-------|------------|------|-------------|--------------|----------|
| **Qwen3-VL-2B** | 2.3B | Dense | 5GB | 2GB | Edge/Mobile |
| **Qwen3-VL-3B** | 3.8B | Dense | 8GB | 3GB | Fast inference |
| **Qwen3-VL-A3B** | 30B (3B active) | MoE | 60GB | 20GB | Quality + Speed |
| **Qwen3-VL-8B** | 8.3B | Dense | 17GB | 6GB | Balanced |
| **Qwen3-VL-32B** | 32B | Dense | 65GB | 20GB | High quality |
| **Qwen3-VL-A30B** | 235B (30B active) | MoE | 470GB | 150GB | Maximum quality |
| **Qwen3-VL-72B** | 72B | Dense | 145GB | 45GB | Frontier |

### Embedding & Reranker Models

| Model | Parameters | Type | VRAM | Purpose |
|-------|------------|------|------|---------|
| **Qwen3-VL-Embedding-2B** | 2B | Dual-tower | 5GB | Fast retrieval |
| **Qwen3-VL-Embedding-8B** | 8B | Dual-tower | 17GB | Quality retrieval |
| **Qwen3-VL-Reranker-2B** | 2B | Cross-encoder | 5GB | Fast reranking |
| **Qwen3-VL-Reranker-8B** | 8B | Cross-encoder | 17GB | Quality reranking |

---

## 1.3 ARCHITECTURE DEEP DIVE

### A. EMBEDDING MODEL (Dual-Tower Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EMBEDDING MODEL ARCHITECTURE                            │
│                        (Dual-Tower / Bi-Encoder)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   QUERY TOWER                           DOCUMENT TOWER                      │
│   ───────────                           ──────────────                      │
│                                                                             │
│   ┌─────────────┐                       ┌─────────────┐                    │
│   │ Screenshot  │                       │ Screenshot  │                    │
│   │ + Text      │                       │ + Text      │                    │
│   └──────┬──────┘                       └──────┬──────┘                    │
│          │                                      │                           │
│          ▼                                      ▼                           │
│   ┌─────────────┐                       ┌─────────────┐                    │
│   │ Vision      │                       │ Vision      │                    │
│   │ Encoder     │                       │ Encoder     │                    │
│   │ (ViT)       │                       │ (ViT)       │                    │
│   └──────┬──────┘                       └──────┬──────┘                    │
│          │                                      │                           │
│          ▼                                      ▼                           │
│   ┌─────────────┐                       ┌─────────────┐                    │
│   │ Qwen3-VL    │                       │ Qwen3-VL    │   SAME WEIGHTS     │
│   │ Transformer │                       │ Transformer │   (shared)         │
│   │ Layers      │                       │ Layers      │                    │
│   └──────┬──────┘                       └──────┬──────┘                    │
│          │                                      │                           │
│          ▼                                      ▼                           │
│   ┌─────────────┐                       ┌─────────────┐                    │
│   │ EOS Token   │                       │ EOS Token   │                    │
│   │ Pooling     │                       │ Pooling     │                    │
│   └──────┬──────┘                       └──────┬──────┘                    │
│          │                                      │                           │
│          ▼                                      ▼                           │
│   ┌─────────────┐                       ┌─────────────┐                    │
│   │ 1024-dim    │                       │ 1024-dim    │                    │
│   │ Vector      │                       │ Vector      │                    │
│   └──────┬──────┘                       └──────┬──────┘                    │
│          │                                      │                           │
│          └──────────────┬───────────────────────┘                           │
│                         │                                                   │
│                         ▼                                                   │
│                  ┌─────────────┐                                            │
│                  │   Cosine    │                                            │
│                  │ Similarity  │ ──────▶  0.87 (similarity score)           │
│                  └─────────────┘                                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  KEY INSIGHT: Query and Document are encoded INDEPENDENTLY                  │
│  - Can pre-compute all document embeddings (offline)                        │
│  - At runtime, only encode query (fast)                                     │
│  - Compare with millions of documents via vector similarity                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  TRAINING: Contrastive Loss (InfoNCE)                                       │
│  - Positive pairs: Query + Correct Document → high similarity               │
│  - Negative pairs: Query + Wrong Document → low similarity                  │
│  - Hard negatives are CRITICAL (similar but wrong)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  SHORTCOMINGS:                                                              │
│  ✗ No cross-attention between query and document                            │
│  ✗ Can't capture fine-grained interactions                                  │
│  ✗ Limited by fixed embedding dimension                                     │
│  ✗ Hard negatives mining is tricky                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. RERANKER MODEL (Cross-Encoder Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RERANKER MODEL ARCHITECTURE                            │
│                        (Cross-Encoder / Single-Tower)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────────────┐                      │
│                    │      CONCATENATED INPUT         │                      │
│                    │  [Query Screenshot + Text]      │                      │
│                    │  [SEP]                          │                      │
│                    │  [Document Text]                │                      │
│                    └───────────────┬─────────────────┘                      │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌─────────────────────────────────┐                      │
│                    │       Vision Encoder (ViT)      │                      │
│                    │    Processes query screenshot   │                      │
│                    └───────────────┬─────────────────┘                      │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌─────────────────────────────────┐                      │
│                    │     Qwen3-VL Transformer        │                      │
│                    │                                 │                      │
│                    │  ┌─────────────────────────┐   │                      │
│                    │  │    FULL CROSS-ATTENTION  │   │                      │
│                    │  │  Query ←──────────→ Doc  │   │                      │
│                    │  │  Every token sees every  │   │                      │
│                    │  │  other token             │   │                      │
│                    │  └─────────────────────────┘   │                      │
│                    │                                 │                      │
│                    └───────────────┬─────────────────┘                      │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌─────────────────────────────────┐                      │
│                    │     Classification Head         │                      │
│                    │                                 │                      │
│                    │  logit_yes = model("yes")      │                      │
│                    │  logit_no  = model("no")       │                      │
│                    │                                 │                      │
│                    │  score = logit_yes - logit_no  │                      │
│                    └───────────────┬─────────────────┘                      │
│                                    │                                        │
│                                    ▼                                        │
│                         ┌───────────────────┐                               │
│                         │  Relevance Score  │                               │
│                         │      0.94         │                               │
│                         └───────────────────┘                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  KEY INSIGHT: Query and Document are processed TOGETHER                     │
│  - Full bidirectional attention between all tokens                          │
│  - Can capture fine-grained semantic relationships                          │
│  - Much more accurate than embedding similarity                             │
│  - But SLOW: must run full model for each (query, doc) pair                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  TRAINING: Binary Cross-Entropy Loss                                        │
│  - Positive: (Query, Correct Doc) → label=1                                 │
│  - Negative: (Query, Wrong Doc) → label=0                                   │
│  - Balanced dataset: 50% positive, 50% negative                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  SHORTCOMINGS:                                                              │
│  ✗ O(n) complexity - must score each document separately                    │
│  ✗ Cannot pre-compute anything                                              │
│  ✗ Slow for large candidate sets                                            │
│  ✗ Memory intensive during inference                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### C. GENERATION MODEL (Decoder-Only LLM)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GENERATION MODEL ARCHITECTURE                           │
│                    (Autoregressive Decoder-Only VLM)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│         ┌─────────────────────────────────────────────────────┐             │
│         │                    INPUT                            │             │
│         │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │             │
│         │  │Screenshot│  │ Retrieved│  │ User Instruction │  │             │
│         │  │ (image)  │  │ Examples │  │    (text)        │  │             │
│         │  └──────────┘  └──────────┘  └──────────────────┘  │             │
│         └────────────────────────┬────────────────────────────┘             │
│                                  │                                          │
│                                  ▼                                          │
│         ┌─────────────────────────────────────────────────────┐             │
│         │              Vision Encoder (ViT)                   │             │
│         │         Patches → Visual Tokens (576+)              │             │
│         └────────────────────────┬────────────────────────────┘             │
│                                  │                                          │
│                                  ▼                                          │
│         ┌─────────────────────────────────────────────────────┐             │
│         │            Qwen3-VL Transformer                     │             │
│         │                                                     │             │
│         │   ┌─────────────────────────────────────────────┐  │             │
│         │   │  DENSE (3B, 8B, 32B, 72B)                   │  │             │
│         │   │  - All parameters active                    │  │             │
│         │   │  - Compute scales with parameters           │  │             │
│         │   └─────────────────────────────────────────────┘  │             │
│         │                       OR                            │             │
│         │   ┌─────────────────────────────────────────────┐  │             │
│         │   │  MoE (A3B, A30B)                            │  │             │
│         │   │  - Only subset of experts active per token  │  │             │
│         │   │  - More parameters, same compute            │  │             │
│         │   │  - A3B: 30B total, 3B active                │  │             │
│         │   │  - A30B: 235B total, 30B active             │  │             │
│         │   └─────────────────────────────────────────────┘  │             │
│         │                                                     │             │
│         └────────────────────────┬────────────────────────────┘             │
│                                  │                                          │
│                                  ▼                                          │
│         ┌─────────────────────────────────────────────────────┐             │
│         │            Autoregressive Generation                │             │
│         │                                                     │             │
│         │   Token 1 → Token 2 → Token 3 → ... → Token N      │             │
│         │   "Click"   "the"    "green"   ...    "</s>"       │             │
│         │                                                     │             │
│         └────────────────────────┬────────────────────────────┘             │
│                                  │                                          │
│                                  ▼                                          │
│         ┌─────────────────────────────────────────────────────┐             │
│         │                    OUTPUT                           │             │
│         │                                                     │             │
│         │   "I'll click the Generate button for you.          │             │
│         │                                                     │             │
│         │    ACTION: CLICK(700, 500)"                         │             │
│         │                                                     │             │
│         └─────────────────────────────────────────────────────┘             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  DENSE vs MoE COMPARISON:                                                   │
│                                                                             │
│  DENSE (32B):                        MoE A3B (30B total, 3B active):        │
│  ✓ Simpler architecture              ✓ 10x more parameters, same speed     │
│  ✓ Easier to train                   ✓ Better generalization               │
│  ✓ More predictable                  ✓ Lower inference cost per token      │
│  ✗ Compute = Parameters              ✗ Complex load balancing              │
│  ✗ Expensive inference               ✗ Expert collapse risk                │
│                                       ✗ Higher memory (stores all experts) │
├─────────────────────────────────────────────────────────────────────────────┤
│  SHORTCOMINGS:                                                              │
│  ✗ Slow autoregressive generation (one token at a time)                    │
│  ✗ Hallucination risk                                                       │
│  ✗ Context length limits                                                    │
│  ✗ Expensive to train and serve                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.4 HOW THEY WORK TOGETHER

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE INFERENCE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USER: "Generate a video of a dancing cat"                                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 1: SCREENSHOT CAPTURE                                           │  │
│  │                                                                       │  │
│  │ Agent captures current screen state → screenshot.png (1024x768)      │  │
│  │ Time: ~50ms                                                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 2: EMBEDDING RETRIEVAL                                          │  │
│  │                                                                       │  │
│  │ Query = Embed(screenshot + "Generate a video of a dancing cat")     │  │
│  │                                                                       │  │
│  │ Compare query vector against 10,000 pre-computed document vectors    │  │
│  │ using approximate nearest neighbor search (FAISS/Annoy)              │  │
│  │                                                                       │  │
│  │ Returns: Top 100 similar past situations                             │  │
│  │ Time: ~10ms                                                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 3: RERANKING                                                    │  │
│  │                                                                       │  │
│  │ For each of 100 candidates:                                          │  │
│  │   score = Reranker(screenshot + query, candidate_action)             │  │
│  │                                                                       │  │
│  │ Sort by score, take top 5                                            │  │
│  │                                                                       │  │
│  │ Returns: 5 most relevant past actions                                │  │
│  │ Time: ~500ms (can be batched)                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 4: GENERATION WITH CONTEXT                                      │  │
│  │                                                                       │  │
│  │ Prompt = f"""                                                        │  │
│  │   <image>{screenshot}</image>                                        │  │
│  │                                                                       │  │
│  │   Here are similar past situations and what worked:                  │  │
│  │   1. {top_5_examples[0]}                                             │  │
│  │   2. {top_5_examples[1]}                                             │  │
│  │   ...                                                                │  │
│  │                                                                       │  │
│  │   User wants: Generate a video of a dancing cat                      │  │
│  │                                                                       │  │
│  │   Based on the examples and current screen, what action to take?     │  │
│  │ """                                                                  │  │
│  │                                                                       │  │
│  │ Output = GenerationModel(Prompt)                                     │  │
│  │ → "Click the prompt field and type 'dancing cat', then click        │  │
│  │    Generate. ACTION: CLICK(400, 300), TYPE(dancing cat)"            │  │
│  │                                                                       │  │
│  │ Time: ~200-2000ms (depends on model size)                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 5: ACTION EXECUTION                                             │  │
│  │                                                                       │  │
│  │ Parse ACTION: CLICK(400, 300)                                        │  │
│  │ Execute via pyautogui/playwright                                     │  │
│  │                                                                       │  │
│  │ Time: ~100ms                                                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 6: LOOP                                                         │  │
│  │                                                                       │  │
│  │ Wait for UI to settle (~500ms)                                       │  │
│  │ Go back to STEP 1                                                    │  │
│  │ Continue until task complete or max steps reached                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  TOTAL LATENCY PER STEP: ~1-3 seconds                                       │
│  TYPICAL TASK (10 steps): ~10-30 seconds                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.5 WHY THIS ARCHITECTURE?

### Why Not Just Use Generation Model Alone?

| Approach | Accuracy | Speed | Cost | Reliability |
|----------|----------|-------|------|-------------|
| Generation only | 50-60% | Fast | $ | Low |
| + SFT | 60-70% | Fast | $ | Medium |
| + Embedding retrieval | 70-75% | +10ms | $$ | High |
| + Reranker | 80-85% | +500ms | $$$ | Very High |
| + DPO | 82-87% | Same | $$$ | Very High |
| + Online RL | 85-90% | Same | $$$$ | Medium |

### The Retrieval Advantage

```
WITHOUT RETRIEVAL:
┌─────────────────────────────────────────────────────────────────┐
│ Screenshot → Generation Model → "Hmm, I think I should..."     │
│                                                                 │
│ The model must MEMORIZE all UI patterns during training        │
│ • Limited by training data                                      │
│ • Fails on slight UI changes                                    │
│ • No way to update without retraining                           │
└─────────────────────────────────────────────────────────────────┘

WITH RETRIEVAL:
┌─────────────────────────────────────────────────────────────────┐
│ Screenshot → Embedding → "I've seen this before!" →            │
│           → Reranker → "These 5 examples are most relevant" →  │
│           → Generation → "Based on example #2, I should..."    │
│                                                                 │
│ The model can LOOK UP similar situations at runtime            │
│ • Add new examples without retraining                           │
│ • Robust to UI variations                                       │
│ • Explainable: "I did this because of example X"               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.6 SHORTCOMINGS AND LIMITATIONS

### Per-Component Limitations

| Component | Limitation | Mitigation |
|-----------|------------|------------|
| **Embedding** | Fixed dimension, lossy compression | Use larger models, instruction-tuning |
| **Embedding** | Hard negative mining is critical | Automated mining pipelines |
| **Reranker** | O(n) scoring, slow for large sets | Limit to top-100 from embedding |
| **Reranker** | Memory intensive | Quantization, batching |
| **Generation** | Hallucination | Retrieval grounding, DPO |
| **Generation** | Slow for long outputs | Speculative decoding |
| **MoE** | Expert load imbalance | Auxiliary losses, expert-choice routing |
| **MoE** | High memory (all experts loaded) | Expert offloading, quantization |

### System-Level Limitations

| Issue | Description | Solution |
|-------|-------------|----------|
| **Latency stacking** | Each component adds latency | Parallel reranking, batching |
| **Error propagation** | Bad retrieval → bad generation | Fallback to generation-only |
| **Index staleness** | Embedding index becomes outdated | Incremental updates, versioning |
| **Context length** | Can't fit many examples | Summarization, compression |
| **UI drift** | UI changes break patterns | Semantic locators, retraining |

---

## 1.7 DECISION GUIDE: WHICH MODELS TO USE?

### For Higgsfield Automation (Your Use Case)

```
RECOMMENDED STACK:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Embedding:   Qwen3-VL-Embedding-8B (best quality)              │
│               or Qwen3-VL-Embedding-2B (if VRAM limited)        │
│                                                                 │
│  Reranker:    Qwen3-VL-Reranker-8B (best quality)              │
│               or Qwen3-VL-Reranker-2B (if latency critical)    │
│                                                                 │
│  Generation:  Qwen3-VL-A3B (MoE - best quality/speed ratio)    │
│               or Qwen3-VL-3B (if single GPU)                   │
│               or Qwen3-VL-32B (if quality is paramount)        │
│                                                                 │
│  Training:    SFT → DPO (offline RL)                            │
│               Add GRPO (online RL) only if DPO plateaus         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Requirements by Configuration

| Configuration | Total VRAM (FP16) | Total VRAM (4-bit) | GPUs Needed |
|---------------|-------------------|--------------------|----|
| Embed-2B + Rerank-2B + Gen-3B | 18GB | 7GB | 1x RTX 4090 |
| Embed-8B + Rerank-8B + Gen-3B | 42GB | 15GB | 1x A100-40GB |
| Embed-8B + Rerank-8B + Gen-A3B | 94GB | 32GB | 1x A100-80GB |
| Embed-8B + Rerank-8B + Gen-32B | 107GB | 38GB | 2x A100-40GB |
| Embed-8B + Rerank-8B + Gen-A30B | 504GB | 165GB | 8x A100-80GB |

---

## NEXT: Part 2 - Training Pipeline (Embedding & Reranker)

Continue to [PART2_TRAINING_EMBEDDING_RERANKER.md](./PART2_TRAINING_EMBEDDING_RERANKER.md)
