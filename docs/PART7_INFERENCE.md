# PART 7: INFERENCE OPTIMIZATION
## vLLM, SGLang, and Production Deployment

---

## 7.1 INFERENCE ENGINES COMPARISON

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE COMPARISON                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  │ Feature              │ vLLM          │ SGLang        │ TensorRT-LLM │  │
│  ├──────────────────────┼───────────────┼───────────────┼──────────────┤  │
│  │ Throughput           │ High          │ Highest       │ High         │  │
│  │ Latency              │ Low           │ Lowest        │ Very Low     │  │
│  │ VLM Support          │ Excellent     │ Good          │ Limited      │  │
│  │ Ease of Use          │ Simple        │ Moderate      │ Complex      │  │
│  │ MoE Support          │ Yes           │ Yes           │ Yes          │  │
│  │ Quantization         │ AWQ/GPTQ/FP8  │ AWQ/GPTQ      │ FP8/INT8     │  │
│  │ Multi-GPU            │ TP/PP         │ TP/PP/DP      │ TP/PP        │  │
│  │ Continuous Batching  │ Yes           │ Yes           │ Yes          │  │
│  │ KV Cache Optimization│ PagedAttention│ RadixAttention│ Custom       │  │
│  │ Speculative Decoding │ Yes           │ Yes           │ Yes          │  │
│  │ Prefix Caching       │ Yes           │ Best (Radix)  │ Limited      │  │
│                                                                             │
│  RECOMMENDATION FOR HIGGSFIELD:                                             │
│  - Primary: vLLM (best VLM support, simpler setup)                         │
│  - Alternative: SGLang (if doing multi-turn conversations)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7.2 vLLM SETUP FOR QWEN3-VL

### Basic Setup

```bash
# Install vLLM
pip install vllm>=0.6.0

# For vision-language models
pip install vllm[vision]
```

### Serving All Models

```python
# VLLM SERVING FOR COMPLETE STACK

from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalInputs
import torch

# ============================================================================
# CONFIGURATION BY MODEL
# ============================================================================

MODEL_CONFIGS = {
    # Embedding models (special handling)
    "embedding-2b": {
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4096,
    },
    "embedding-8b": {
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
    },

    # Reranker models
    "reranker-2b": {
        "model": "Qwen/Qwen3-VL-Reranker-2B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 2048,
    },
    "reranker-8b": {
        "model": "Qwen/Qwen3-VL-Reranker-8B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,
    },

    # Generation models - Dense
    "gen-3b": {
        "model": "Qwen/Qwen3-VL-3B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
    },
    "gen-8b": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 8192,
    },
    "gen-32b": {
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "tensor_parallel_size": 2,  # Needs 2 GPUs
        "gpu_memory_utilization": 0.95,
        "max_model_len": 8192,
    },

    # Generation models - MoE
    "gen-a3b": {
        "model": "Qwen/Qwen3-VL-A3B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 8192,
        # MoE specific
        "enable_expert_parallel": False,  # TP handles it
    },
    "gen-a30b": {
        "model": "Qwen/Qwen3-VL-A30B-Instruct",
        "tensor_parallel_size": 8,  # Needs 8 GPUs
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 8192,
    },
}

# ============================================================================
# VLLM SERVER
# ============================================================================

class VLLMServer:
    """vLLM server for Qwen3-VL models."""

    def __init__(self, model_name: str, quantization: str = None):
        config = MODEL_CONFIGS[model_name]

        self.llm = LLM(
            model=config["model"],
            tensor_parallel_size=config["tensor_parallel_size"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            max_model_len=config["max_model_len"],
            trust_remote_code=True,
            # Quantization
            quantization=quantization,  # "awq", "gptq", "fp8"
            # Performance
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            # For VLMs
            limit_mm_per_prompt={"image": 5},  # Max 5 images per prompt
        )

        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompt: str,
        images: list = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        """Generate response."""

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Prepare multimodal inputs
        if images:
            mm_data = {"image": images}
            outputs = self.llm.generate(
                [{"prompt": prompt, "multi_modal_data": mm_data}],
                sampling_params,
            )
        else:
            outputs = self.llm.generate([prompt], sampling_params)

        return outputs[0].outputs[0].text

    def batch_generate(
        self,
        prompts: list,
        images_list: list = None,
        max_tokens: int = 256,
    ):
        """Batch generation for throughput."""

        sampling_params = SamplingParams(max_tokens=max_tokens)

        if images_list:
            inputs = [
                {"prompt": p, "multi_modal_data": {"image": imgs}}
                for p, imgs in zip(prompts, images_list)
            ]
        else:
            inputs = prompts

        outputs = self.llm.generate(inputs, sampling_params)

        return [o.outputs[0].text for o in outputs]


# ============================================================================
# COMMAND LINE SERVING
# ============================================================================

# Serve generation model
"""
vllm serve Qwen/Qwen3-VL-A3B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --trust-remote-code \
    --port 8000

# With quantization
vllm serve Qwen/Qwen3-VL-A3B-Instruct \
    --quantization awq \
    --tensor-parallel-size 1 \
    --port 8000

# Multi-GPU (32B model)
vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000
"""
```

---

## 7.3 OPTIMIZATION TECHNIQUES

### Continuous Batching

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS BATCHING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STATIC BATCHING (traditional):                                             │
│  ────────────────────────────                                               │
│  Wait for batch to fill → Process all → Return all                         │
│                                                                             │
│  Request 1: ████████████████                                               │
│  Request 2: ████████████████████████████                                   │
│  Request 3: ████████                                                        │
│             └─────── Wait for longest ───────┘                             │
│                                                                             │
│  Problem: Short requests wait for long requests                            │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  CONTINUOUS BATCHING (vLLM/SGLang):                                        │
│  ─────────────────────────────────                                          │
│  Process tokens as they complete, add new requests immediately             │
│                                                                             │
│  Request 1: ████████████████ ✓ (done, freed)                               │
│  Request 2: ████████████████████████████ ✓                                 │
│  Request 3: ████████ ✓         New request 4: ████████████                 │
│                                New request 5: ████                          │
│             └─── No waiting, continuous processing ───┘                    │
│                                                                             │
│  Benefit: ~2-3x throughput improvement                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### PagedAttention (vLLM)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PAGED ATTENTION                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: KV cache wastes memory                                           │
│                                                                             │
│  Traditional: Pre-allocate max_seq_len for each request                    │
│  Request 1: [████████____________________] (20% used, 80% wasted)          │
│  Request 2: [██████████████______________] (40% used, 60% wasted)          │
│                                                                             │
│  PAGED ATTENTION: Allocate memory in pages, like OS virtual memory        │
│                                                                             │
│  Physical memory: [Page 0][Page 1][Page 2][Page 3][Page 4]...              │
│  Request 1:       [Page 0 → Page 3]                                        │
│  Request 2:       [Page 1 → Page 2 → Page 4]                               │
│                                                                             │
│  Only allocate pages as needed, reuse freed pages                          │
│                                                                             │
│  Benefit: ~2-4x more requests in same memory                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RadixAttention (SGLang)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RADIX ATTENTION (SGLang)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: Multi-turn conversations recompute shared prefix                 │
│                                                                             │
│  Turn 1: [System prompt + User 1] → Response 1                             │
│  Turn 2: [System prompt + User 1 + Response 1 + User 2] → Response 2       │
│          ^^^^^^^^^^^^^^^^^^^^^^^^ Recomputed!                              │
│                                                                             │
│  RADIX TREE: Store KV cache by prefix, share across requests               │
│                                                                             │
│                    [System prompt]                                          │
│                          │                                                  │
│              ┌───────────┴───────────┐                                     │
│              │                       │                                      │
│        [User 1]                 [User 2]                                   │
│              │                       │                                      │
│       [Response 1]            [Response 2]                                 │
│              │                                                              │
│        [User 3]                                                             │
│                                                                             │
│  Shared prefixes are computed ONCE, cached, and reused                     │
│                                                                             │
│  Benefit: 3-5x speedup for multi-turn, 60%+ cache hit rate                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Speculative Decoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPECULATIVE DECODING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: Autoregressive decoding is sequential (slow)                     │
│                                                                             │
│  Token 1 → Token 2 → Token 3 → Token 4 → ...                               │
│  Each token requires full model forward pass                               │
│                                                                             │
│  SOLUTION: Use small "draft" model to speculate multiple tokens           │
│                                                                             │
│  1. Draft model predicts K tokens ahead (fast, small model)               │
│  2. Target model verifies all K tokens in parallel (one pass)             │
│  3. Accept matching tokens, reject and regenerate from first mismatch     │
│                                                                             │
│  Draft: "The cat sat on the mat"                                           │
│  Verify: "The cat sat on the rug"  ← "rug" != "mat", reject from here     │
│  Accept: "The cat sat on the" (5 tokens in 2 forward passes!)             │
│                                                                             │
│  Speedup: 1.5-2x for well-matched draft models                            │
│                                                                             │
│  vLLM usage:                                                                │
│  vllm serve Qwen/Qwen3-VL-A3B-Instruct \                                   │
│      --speculative-model Qwen/Qwen3-VL-0.6B-Instruct \                     │
│      --num-speculative-tokens 5                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7.4 QUANTIZATION

### Quantization Options

| Method | Bits | Speed Impact | Quality Impact | Memory Savings |
|--------|------|--------------|----------------|----------------|
| **FP16** | 16 | Baseline | Baseline | 1x |
| **BF16** | 16 | Same | Same | 1x |
| **FP8** | 8 | +10-20% | <1% loss | 2x |
| **AWQ** | 4 | +20-30% | 1-2% loss | 4x |
| **GPTQ** | 4 | +15-25% | 1-3% loss | 4x |
| **GGUF** | 2-8 | Varies | Varies | 2-8x |

### Recommended Quantization by Use Case

```python
# QUANTIZATION RECOMMENDATIONS

QUANTIZATION_GUIDE = {
    # Embedding models: Quality critical, use FP16 or FP8
    "embedding": {
        "production": "fp8",   # Best quality/speed tradeoff
        "budget": "awq",       # 4-bit if memory constrained
    },

    # Reranker models: Quality critical, use FP16 or FP8
    "reranker": {
        "production": "fp8",
        "budget": "awq",
    },

    # Generation models: Can tolerate more quantization
    "generation": {
        "quality_first": "fp8",
        "balanced": "awq",      # Recommended for most cases
        "throughput_first": "awq",
    },
}

# vLLM with quantization
"""
# AWQ quantization (recommended for 4-bit)
vllm serve Qwen/Qwen3-VL-A3B-Instruct-AWQ \
    --quantization awq \
    --port 8000

# FP8 quantization (best for Hopper GPUs)
vllm serve Qwen/Qwen3-VL-A3B-Instruct \
    --quantization fp8 \
    --port 8000

# Dynamic quantization at load time
vllm serve Qwen/Qwen3-VL-A3B-Instruct \
    --quantization fp8_e4m3 \
    --kv-cache-dtype fp8_e4m3 \
    --port 8000
"""
```

---

## 7.5 PARALLELISM STRATEGIES

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLELISM FOR MULTI-GPU                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TENSOR PARALLELISM (TP):                                                   │
│  ───────────────────────                                                    │
│  Split each layer across GPUs                                              │
│                                                                             │
│  GPU 0: [Layer weights 0-25%] ──┐                                          │
│  GPU 1: [Layer weights 25-50%]  ├── All-reduce after each layer           │
│  GPU 2: [Layer weights 50-75%]  │                                          │
│  GPU 3: [Layer weights 75-100%]─┘                                          │
│                                                                             │
│  Pros: Reduces latency (all GPUs work together per token)                  │
│  Cons: High communication (NVLink required)                                │
│  Use when: GPUs have fast interconnect (same node, NVLink)                 │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  PIPELINE PARALLELISM (PP):                                                 │
│  ─────────────────────────                                                  │
│  Split model into stages, each GPU handles some layers                     │
│                                                                             │
│  GPU 0: [Layers 0-7]  ──► GPU 1: [Layers 8-15]  ──►                        │
│  GPU 2: [Layers 16-23] ──► GPU 3: [Layers 24-31]                           │
│                                                                             │
│  Pros: Less communication (only between stages)                            │
│  Cons: Pipeline bubbles, higher latency                                    │
│  Use when: GPUs are across nodes (slow interconnect)                       │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  EXPERT PARALLELISM (EP) - for MoE:                                        │
│  ─────────────────────────────────                                          │
│  Split experts across GPUs                                                 │
│                                                                             │
│  GPU 0: [Experts 0-15]                                                     │
│  GPU 1: [Experts 16-31]                                                    │
│  GPU 2: [Experts 32-47]                                                    │
│  GPU 3: [Experts 48-63]                                                    │
│                                                                             │
│  Pros: Good for MoE models, less memory per GPU                           │
│  Cons: Load balancing challenges                                           │
│  Use when: Running large MoE models (A30B)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recommended Configurations

| Model | GPUs | TP | PP | Memory/GPU |
|-------|------|----|----|------------|
| 3B | 1 | 1 | 1 | 8GB |
| A3B | 1 | 1 | 1 | 60GB |
| 8B | 1 | 1 | 1 | 17GB |
| 32B | 2 | 2 | 1 | 33GB |
| 32B | 4 | 4 | 1 | 17GB |
| A30B | 4 | 4 | 1 | 120GB |
| A30B | 8 | 8 | 1 | 60GB |
| 72B | 4 | 4 | 1 | 37GB |
| 72B | 8 | 4 | 2 | 20GB |

```bash
# Examples

# 32B on 2x A100-80GB
vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --tensor-parallel-size 2

# A30B on 8x A100-80GB
vllm serve Qwen/Qwen3-VL-A30B-Instruct \
    --tensor-parallel-size 8

# 72B on 2 nodes × 4 GPUs
vllm serve Qwen/Qwen3-VL-72B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

---

## 7.6 LATENCY OPTIMIZATION

### Latency Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE LATENCY BREAKDOWN                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PREFILL PHASE (process input):                                            │
│  ─────────────────────────────                                              │
│  - Tokenize input: ~1ms                                                    │
│  - Process image (ViT): ~20-50ms                                           │
│  - Compute KV cache for all input tokens: 50-500ms                        │
│  - Total: 70-550ms                                                          │
│                                                                             │
│  DECODE PHASE (generate output):                                           │
│  ──────────────────────────────                                             │
│  - Per token: 5-20ms (memory bound)                                        │
│  - 100 tokens: 500-2000ms                                                   │
│  - Total: 500-2000ms                                                        │
│                                                                             │
│  TOTAL TIME = Prefill + Decode                                             │
│  Typical: 600-2500ms for complete response                                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OPTIMIZATION TARGETS:                                                      │
│                                                                             │
│  1. PREFILL: Chunked prefill, tensor parallelism                           │
│  2. DECODE: Speculative decoding, continuous batching                      │
│  3. BOTH: Quantization, better hardware                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Optimization Checklist

```python
# LATENCY OPTIMIZATION CHECKLIST

OPTIMIZATIONS = {
    "essential": [
        "Enable CUDA graphs (automatic in vLLM)",
        "Use FlashAttention 2/3",
        "Enable prefix caching",
        "Set appropriate max_model_len (not too large)",
    ],
    "recommended": [
        "Use FP8 quantization on Hopper GPUs",
        "Enable chunked prefill for long inputs",
        "Use tensor parallelism if >1 GPU available",
        "Tune gpu_memory_utilization (0.9-0.95)",
    ],
    "advanced": [
        "Speculative decoding with draft model",
        "Custom CUDA kernels for specific ops",
        "KV cache compression",
        "Request-level scheduling optimization",
    ],
}

# Optimal vLLM configuration for latency
"""
vllm serve Qwen/Qwen3-VL-A3B-Instruct \
    --quantization fp8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --enforce-eager false \
    --port 8000
"""
```

---

## 7.7 THROUGHPUT OPTIMIZATION

### Batching Configuration

```python
# THROUGHPUT OPTIMIZATION

# Key parameters for throughput
THROUGHPUT_CONFIG = {
    "max_num_seqs": 256,          # Max concurrent sequences
    "max_num_batched_tokens": 32768,  # Max tokens per batch
    "gpu_memory_utilization": 0.95,   # Use most GPU memory
    "swap_space": 4,              # GB of CPU swap for overflow
    "enable_chunked_prefill": True,   # Better batching
}

# Throughput-optimized vLLM command
"""
vllm serve Qwen/Qwen3-VL-A3B-Instruct \
    --quantization awq \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.95 \
    --swap-space 4 \
    --enable-chunked-prefill \
    --disable-log-requests \
    --port 8000
"""
```

### Expected Throughput by Model

| Model | GPU | Quantization | Tokens/sec | Requests/sec |
|-------|-----|--------------|------------|--------------|
| 3B | 1xA100 | FP16 | 3000 | 15-20 |
| 3B | 1xA100 | AWQ | 4500 | 20-30 |
| A3B | 1xA100 | FP16 | 2500 | 12-15 |
| A3B | 1xA100 | AWQ | 3800 | 18-25 |
| 32B | 2xA100 | AWQ | 1200 | 6-10 |
| A30B | 8xA100 | AWQ | 800 | 4-8 |

---

## 7.8 PRODUCTION DEPLOYMENT

### Docker Deployment

```dockerfile
# Dockerfile for vLLM + Qwen3-VL
FROM vllm/vllm-openai:latest

# Install additional dependencies
RUN pip install pillow qwen-vl-utils

# Copy custom configuration
COPY config.yaml /app/config.yaml

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen3-VL-A3B-Instruct", \
     "--tensor-parallel-size", "1", \
     "--port", "8000"]
```

```yaml
# docker-compose.yaml
version: '3.8'
services:
  vllm-generation:
    image: vllm-qwen3vl:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  embedding-server:
    image: embedding-qwen3vl:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
    ports:
      - "8001:8001"

  reranker-server:
    image: reranker-qwen3vl:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1  # Can share with embedding
    ports:
      - "8002:8002"
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3vl-generation
spec:
  replicas: 2
  selector:
    matchLabels:
      app: qwen3vl-generation
  template:
    metadata:
      labels:
        app: qwen3vl-generation
    spec:
      containers:
      - name: vllm
        image: vllm-qwen3vl:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3vl-generation-svc
spec:
  selector:
    app: qwen3vl-generation
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

---

## NEXT: Part 8 - GPU Specifications & Optimization

Continue to [PART8_GPU_SPECS.md](./PART8_GPU_SPECS.md)
