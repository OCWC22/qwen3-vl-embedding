# PART 6: EVALUATION & BENCHMARKS
## How to Measure Model Performance

---

## 6.1 EVALUATION HIERARCHY

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION LEVELS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 1: COMPONENT METRICS                                                 │
│  ─────────────────────────                                                  │
│  Evaluate each model independently                                          │
│  - Embedding: Recall@K, MRR                                                │
│  - Reranker: Accuracy, NDCG                                                │
│  - Generation: Perplexity, Format Compliance                               │
│                                                                             │
│  LEVEL 2: PIPELINE METRICS                                                  │
│  ────────────────────────                                                   │
│  Evaluate end-to-end retrieval + generation                               │
│  - Action Accuracy                                                          │
│  - Retrieval-augmented metrics                                             │
│                                                                             │
│  LEVEL 3: TASK METRICS                                                      │
│  ──────────────────────                                                     │
│  Evaluate on actual tasks                                                   │
│  - Task Success Rate                                                        │
│  - Steps to Completion                                                      │
│  - Error Recovery Rate                                                      │
│                                                                             │
│  LEVEL 4: BENCHMARK METRICS                                                 │
│  ────────────────────────                                                   │
│  Compare against standard benchmarks                                        │
│  - REAL Bench (AGI Inc)                                                     │
│  - OSWorld                                                                  │
│  - WebArena                                                                 │
│  - ScreenSpot                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6.2 COMPONENT METRICS

### Embedding Model Metrics

| Metric | Formula | What it Measures | Target |
|--------|---------|------------------|--------|
| **Recall@K** | (correct in top-K) / total | Coverage | R@10 > 85% |
| **MRR** | mean(1/rank of correct) | Ranking quality | > 0.6 |
| **NDCG@K** | DCG@K / ideal DCG@K | Graded relevance | > 0.7 |
| **Precision@K** | (correct in top-K) / K | Precision | P@5 > 40% |

```python
def evaluate_embedding(model, test_data, index, k_values=[1, 5, 10, 100]):
    """Evaluate embedding model on retrieval task."""

    results = {f"recall@{k}": 0 for k in k_values}
    results["mrr"] = 0

    for example in test_data:
        # Encode query
        query_embedding = model.encode(example["query"])

        # Search index
        scores, indices = index.search(query_embedding, max(k_values))

        # Find rank of correct document
        correct_id = example["correct_doc_id"]
        if correct_id in indices:
            rank = list(indices).index(correct_id) + 1
            results["mrr"] += 1 / rank

            for k in k_values:
                if rank <= k:
                    results[f"recall@{k}"] += 1

    # Normalize
    n = len(test_data)
    for key in results:
        results[key] /= n

    return results
```

### Reranker Model Metrics

| Metric | Formula | What it Measures | Target |
|--------|---------|------------------|--------|
| **Accuracy** | correct / total | Classification | > 85% |
| **AUC-ROC** | Area under ROC curve | Discrimination | > 0.9 |
| **F1 Score** | 2 × (P × R) / (P + R) | Balance | > 0.85 |
| **MRR Lift** | MRR_after / MRR_before | Improvement | > 1.3 |

```python
def evaluate_reranker(model, test_data, embedding_model, k=100):
    """Evaluate reranker on reranking task."""

    results = {
        "accuracy": 0,
        "mrr_before": 0,
        "mrr_after": 0,
    }

    for example in test_data:
        # Get top-K from embedding
        candidates = embedding_model.retrieve(example["query"], k=k)

        # Rerank
        reranked = model.rerank(example["query"], candidates)

        # Find correct document
        correct_id = example["correct_doc_id"]

        # Before reranking
        if correct_id in [c["id"] for c in candidates]:
            rank_before = [c["id"] for c in candidates].index(correct_id) + 1
            results["mrr_before"] += 1 / rank_before

        # After reranking
        if correct_id in [c["id"] for c in reranked]:
            rank_after = [c["id"] for c in reranked].index(correct_id) + 1
            results["mrr_after"] += 1 / rank_after

    # Normalize and compute lift
    n = len(test_data)
    results["mrr_before"] /= n
    results["mrr_after"] /= n
    results["mrr_lift"] = results["mrr_after"] / results["mrr_before"]

    return results
```

### Generation Model Metrics

| Metric | Formula | What it Measures | Target |
|--------|---------|------------------|--------|
| **Perplexity** | exp(avg NLL) | Language modeling | < 5 |
| **Format Compliance** | has ACTION / total | Output format | > 95% |
| **Action Accuracy** | correct action / total | Action correctness | > 85% |
| **Coordinate Error** | mean |pred - actual| | Click precision | < 50px |

---

## 6.3 STANDARD BENCHMARKS

### REAL Bench (AGI Inc)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REAL BENCH - AGI Inc                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHAT: 112 tasks across website replicas                                   │
│                                                                             │
│  DOMAINS:                                                                   │
│  - Amazon shopping                                                          │
│  - DoorDash ordering                                                        │
│  - Google Flights booking                                                   │
│  - Gmail composition                                                        │
│  - Calendar scheduling                                                      │
│                                                                             │
│  TASK TYPES:                                                                │
│  - Information retrieval ("Find cheapest flight to NYC")                   │
│  - Form filling ("Book a table for 4 at 7pm")                             │
│  - Navigation ("Go to order history")                                      │
│  - Transaction ("Add item to cart and checkout")                          │
│                                                                             │
│  SCORING:                                                                   │
│  - Task Success Rate (binary: completed or not)                            │
│  - Partial credit for progress                                             │
│                                                                             │
│  CURRENT SCORES (2025):                                                     │
│  │ Model               │ Score │                                           │
│  ├─────────────────────┼───────┤                                           │
│  │ AGI Agent-0         │ 45%   │ (SOTA)                                    │
│  │ Claude Computer Use │ 38%   │                                           │
│  │ GPT-4V + Actions    │ 32%   │                                           │
│  │ Gemini Pro Vision   │ 28%   │                                           │
│                                                                             │
│  URL: https://www.agi-inc.io/research/real-bench                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### OSWorld

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OSWORLD - Desktop Automation                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHAT: 369 tasks in real desktop environments                              │
│                                                                             │
│  ENVIRONMENTS:                                                              │
│  - Ubuntu (Linux)                                                           │
│  - Windows                                                                  │
│  - macOS                                                                    │
│                                                                             │
│  APPLICATIONS:                                                              │
│  - VS Code                                                                  │
│  - LibreOffice                                                              │
│  - Chrome/Firefox                                                           │
│  - File managers                                                            │
│  - Terminal                                                                 │
│                                                                             │
│  TASK EXAMPLES:                                                             │
│  - "Create a new Python file and write a hello world program"             │
│  - "Find all PDF files modified in the last week"                         │
│  - "Change the desktop wallpaper to the downloaded image"                 │
│                                                                             │
│  CURRENT SCORES:                                                            │
│  │ Model            │ Score  │                                             │
│  ├──────────────────┼────────┤                                             │
│  │ AGI Agent        │ 76.26% │ (World Record)                              │
│  │ Claude 3.5       │ 22.0%  │                                             │
│  │ GPT-4V           │ 12.24% │                                             │
│  │ Gemini Pro       │ 4.79%  │                                             │
│                                                                             │
│  URL: https://os-world.github.io/                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### WebArena

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WEBARENA - Web Automation                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHAT: 812 tasks across 4 web domains                                      │
│                                                                             │
│  DOMAINS:                                                                   │
│  1. E-commerce (OneStopShop)                                               │
│  2. Forums (Reddit-like)                                                    │
│  3. CMS (GitLab-like)                                                      │
│  4. Maps (OpenStreetMap)                                                   │
│                                                                             │
│  TASK TYPES:                                                                │
│  - Information seeking                                                      │
│  - Site navigation                                                          │
│  - Content & Config                                                         │
│                                                                             │
│  CURRENT SCORES:                                                            │
│  │ Model              │ Score  │                                           │
│  ├────────────────────┼────────┤                                           │
│  │ AWM + GPT-4o       │ 61.7%  │ (Record)                                  │
│  │ SteP + GPT-4       │ 35.8%  │                                           │
│  │ GPT-4V baseline    │ 14.41% │                                           │
│                                                                             │
│  URL: https://webarena.dev/                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ScreenSpot

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCREENSPOT - GUI Grounding                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHAT: 1.2K GUI screenshots with grounding annotations                     │
│                                                                             │
│  TASK: Given screenshot + instruction, locate UI element                   │
│                                                                             │
│  EXAMPLE:                                                                   │
│  Screenshot: [Gmail interface]                                              │
│  Instruction: "Click the compose button"                                   │
│  Output: Bounding box of compose button                                    │
│                                                                             │
│  METRICS:                                                                   │
│  - Click Accuracy (point in element)                                       │
│  - IoU (Intersection over Union of bounding boxes)                        │
│                                                                             │
│  RELEVANCE: Tests embedding/retrieval grounding ability                    │
│                                                                             │
│  URL: https://github.com/showlab/ScreenSpot                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MTEB & MMEB (Embedding Benchmarks)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MTEB / MMEB - Embedding Benchmarks                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MTEB (Massive Text Embedding Benchmark):                                   │
│  - 56 datasets across 8 task types                                         │
│  - Retrieval, Classification, Clustering, etc.                             │
│  - Primary benchmark for text embeddings                                   │
│                                                                             │
│  MMEB (Massive Multimodal Embedding Benchmark):                            │
│  - 36 datasets for vision-language embeddings                              │
│  - Image-text retrieval, VQA, Visual grounding                            │
│  - Primary benchmark for VL embeddings                                     │
│                                                                             │
│  M-BEIR (Multimodal BEIR):                                                 │
│  - 8 tasks, 16 datasets                                                    │
│  - Cross-modal retrieval                                                    │
│                                                                             │
│  USE THESE TO EVALUATE: Embedding model quality                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6.4 CUSTOM BENCHMARK FOR HIGGSFIELD

Create your own benchmark specific to Higgsfield:

```python
# HIGGSFIELD BENCHMARK

HIGGSFIELD_BENCHMARK = {
    "name": "Higgsfield-Bench-v1",
    "tasks": [
        # AUTHENTICATION (10 tasks)
        {
            "id": "auth_001",
            "category": "authentication",
            "instruction": "Log in with email test@example.com and password Test123!",
            "start_url": "https://higgsfield.ai/login",
            "success_criteria": "User is logged in and sees dashboard",
            "max_steps": 5,
        },
        {
            "id": "auth_002",
            "category": "authentication",
            "instruction": "Sign up for a new account",
            "start_url": "https://higgsfield.ai/login",
            "success_criteria": "Registration form submitted",
            "max_steps": 8,
        },
        # ... more auth tasks

        # MODEL SELECTION (15 tasks)
        {
            "id": "model_001",
            "category": "model_selection",
            "instruction": "Select the Animate model for image animation",
            "start_url": "https://higgsfield.ai/workspace",
            "success_criteria": "Animate model is selected",
            "max_steps": 3,
        },
        # ... more model tasks

        # UPLOAD (10 tasks)
        {
            "id": "upload_001",
            "category": "upload",
            "instruction": "Upload the image cat.jpg from downloads",
            "start_url": "https://higgsfield.ai/workspace",
            "success_criteria": "Image is uploaded and preview shown",
            "max_steps": 4,
        },
        # ... more upload tasks

        # GENERATION (20 tasks)
        {
            "id": "gen_001",
            "category": "generation",
            "instruction": "Generate a 5 second video at 720p",
            "start_url": "https://higgsfield.ai/workspace",
            "precondition": "Image already uploaded, model selected",
            "success_criteria": "Video generation started",
            "max_steps": 5,
        },
        # ... more generation tasks

        # END-TO-END (10 tasks)
        {
            "id": "e2e_001",
            "category": "end_to_end",
            "instruction": "Create a dancing cat video using the Animate model",
            "start_url": "https://higgsfield.ai/login",
            "success_criteria": "Video is generated and downloadable",
            "max_steps": 15,
        },
        # ... more e2e tasks

        # ERROR RECOVERY (5 tasks)
        {
            "id": "error_001",
            "category": "error_recovery",
            "instruction": "Handle the login error and retry",
            "start_url": "https://higgsfield.ai/login",
            "precondition": "Previous login failed",
            "success_criteria": "Successfully logged in after retry",
            "max_steps": 6,
        },
    ],
    "total_tasks": 70,
    "scoring": {
        "task_success": 1.0,
        "partial_progress": 0.5,
        "failed": 0.0,
    },
}

def run_benchmark(agent, benchmark):
    """Run benchmark and compute scores."""

    results = {
        "total_score": 0,
        "tasks_completed": 0,
        "tasks_partial": 0,
        "tasks_failed": 0,
        "by_category": {},
        "avg_steps": 0,
    }

    total_steps = 0

    for task in benchmark["tasks"]:
        # Run task
        task_result = run_task(agent, task)

        # Score
        if task_result["success"]:
            score = benchmark["scoring"]["task_success"]
            results["tasks_completed"] += 1
        elif task_result["progress"] > 0.5:
            score = benchmark["scoring"]["partial_progress"]
            results["tasks_partial"] += 1
        else:
            score = benchmark["scoring"]["failed"]
            results["tasks_failed"] += 1

        results["total_score"] += score
        total_steps += task_result["steps_taken"]

        # By category
        cat = task["category"]
        if cat not in results["by_category"]:
            results["by_category"][cat] = {"score": 0, "count": 0}
        results["by_category"][cat]["score"] += score
        results["by_category"][cat]["count"] += 1

    # Normalize
    results["total_score"] /= benchmark["total_tasks"]
    results["total_score"] *= 100  # Percentage
    results["avg_steps"] = total_steps / benchmark["total_tasks"]

    for cat in results["by_category"]:
        results["by_category"][cat]["score"] /= results["by_category"][cat]["count"]
        results["by_category"][cat]["score"] *= 100

    return results
```

---

## 6.5 EVALUATION TARGETS

### By Training Stage

| Stage | Primary Metric | Target | Pass Threshold |
|-------|----------------|--------|----------------|
| Embedding | Recall@100 | >95% | >90% |
| Reranker | MRR Lift | >1.3x | >1.2x |
| SFT | Format Compliance | >95% | >90% |
| DPO | Action Accuracy | >85% | >80% |
| GRPO | Task Success | >80% | >70% |

### By Benchmark

| Benchmark | Metric | Target | World Record |
|-----------|--------|--------|--------------|
| REAL Bench | Success Rate | >50% | 45% |
| OSWorld | Success Rate | >30% | 76% |
| WebArena | Success Rate | >40% | 62% |
| ScreenSpot | Click Acc | >85% | ~90% |
| Higgsfield-Bench | Success Rate | >75% | N/A (yours) |

---

## 6.6 EVALUATION PIPELINE

```python
# FULL EVALUATION PIPELINE

class EvaluationPipeline:
    """Complete evaluation for Higgsfield agent."""

    def __init__(self, agent, benchmarks):
        self.agent = agent
        self.benchmarks = benchmarks

    def run_all(self):
        """Run all evaluations."""

        results = {}

        # 1. Component evaluations
        print("=" * 60)
        print("COMPONENT EVALUATION")
        print("=" * 60)

        results["embedding"] = self.eval_embedding()
        results["reranker"] = self.eval_reranker()
        results["generation"] = self.eval_generation()

        # 2. Pipeline evaluation
        print("=" * 60)
        print("PIPELINE EVALUATION")
        print("=" * 60)

        results["pipeline"] = self.eval_pipeline()

        # 3. Benchmark evaluations
        print("=" * 60)
        print("BENCHMARK EVALUATION")
        print("=" * 60)

        for bench_name, bench in self.benchmarks.items():
            results[bench_name] = self.run_benchmark(bench)

        # 4. Summary
        self.print_summary(results)

        return results

    def print_summary(self, results):
        """Print evaluation summary."""

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print("\nComponent Metrics:")
        print(f"  Embedding Recall@100: {results['embedding']['recall@100']:.1%}")
        print(f"  Reranker MRR Lift:    {results['reranker']['mrr_lift']:.2f}x")
        print(f"  Generation Format:    {results['generation']['format_compliance']:.1%}")

        print("\nBenchmark Scores:")
        for bench in self.benchmarks:
            if bench in results:
                print(f"  {bench}: {results[bench]['total_score']:.1f}%")

        # Overall grade
        overall = sum(r.get("total_score", 0) for r in results.values()
                      if isinstance(r, dict) and "total_score" in r)
        overall /= len([r for r in results.values()
                        if isinstance(r, dict) and "total_score" in r])

        print(f"\nOverall Score: {overall:.1f}%")

        if overall >= 80:
            print("Grade: A (Excellent)")
        elif overall >= 70:
            print("Grade: B (Good)")
        elif overall >= 60:
            print("Grade: C (Acceptable)")
        else:
            print("Grade: D (Needs Improvement)")
```

---

## NEXT: Part 7 - Inference Optimization (vLLM & SGLang)

Continue to [PART7_INFERENCE.md](./PART7_INFERENCE.md)
