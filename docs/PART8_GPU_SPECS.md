# PART 8: GPU SPECIFICATIONS & OPTIMIZATION
## A100, H100, B200, GB200, GB300 Complete Comparison

---

## 8.1 GPU COMPARISON TABLE

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              NVIDIA DATA CENTER GPU COMPARISON                                       │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────────────────┤
│ Spec         │ A100         │ H100         │ B200         │ GB200        │ GB300                    │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────┤
│ Architecture │ Ampere       │ Hopper       │ Blackwell    │ Blackwell    │ Blackwell Ultra          │
│ Released     │ 2020         │ 2022         │ 2024         │ 2024         │ 2025                     │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────┤
│ VRAM         │ 40GB / 80GB  │ 80GB         │ 192GB        │ 192GB        │ 288GB                    │
│ Memory Type  │ HBM2e        │ HBM3         │ HBM3e        │ HBM3e        │ HBM3e                    │
│ Bandwidth    │ 2 TB/s       │ 3.35 TB/s    │ 8 TB/s       │ 8 TB/s       │ 8 TB/s                   │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────┤
│ FP16 TFLOPS  │ 312          │ 990          │ 2,250        │ 2,500        │ ~3,000                   │
│ FP8 TFLOPS   │ N/A          │ 1,980        │ 4,500        │ 5,000        │ ~6,000                   │
│ INT8 TOPS    │ 624          │ 1,980        │ 4,500        │ 5,000        │ ~6,000                   │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────┤
│ NVLink       │ 600 GB/s     │ 900 GB/s     │ 1.8 TB/s     │ 1.8 TB/s     │ 1.8 TB/s                 │
│ NVLink GPUs  │ Up to 8      │ Up to 8      │ Up to 72     │ Up to 72     │ Up to 72                 │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────┤
│ TDP          │ 400W         │ 700W         │ 1000W        │ 1200W        │ ~1400W                   │
│ Cooling      │ Air/Liquid   │ Air/Liquid   │ Liquid       │ Liquid       │ Liquid                   │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────┤
│ Price (Est.) │ $10-15K      │ $25-40K      │ $30-50K      │ $60-80K      │ TBD                      │
│ Availability │ Excellent    │ Good         │ Limited      │ Very Limited │ 2025 H2                  │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────────────────┘
```

---

## 8.2 DETAILED GPU PROFILES

### A100 (Ampere)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA A100                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VARIANTS:                                                                  │
│  - A100-40GB: 40GB HBM2e, 1.6 TB/s bandwidth                               │
│  - A100-80GB: 80GB HBM2e, 2.0 TB/s bandwidth (RECOMMENDED)                 │
│  - A100-80GB SXM: 80GB, NVLink, for DGX/HGX systems                       │
│                                                                             │
│  BEST FOR:                                                                  │
│  ✓ Most cost-effective for training                                        │
│  ✓ Widely available (cloud & on-prem)                                      │
│  ✓ Good balance of memory and compute                                      │
│  ✓ Mature ecosystem and tooling                                            │
│                                                                             │
│  LIMITATIONS:                                                               │
│  ✗ No FP8 support (Hopper+ only)                                           │
│  ✗ Lower memory bandwidth than newer GPUs                                  │
│  ✗ Limited to 8-GPU NVLink domain                                          │
│                                                                             │
│  QWEN3-VL CAPACITY:                                                         │
│  │ Model         │ A100-40GB   │ A100-80GB   │                             │
│  ├───────────────┼─────────────┼─────────────┤                             │
│  │ 3B (FP16)     │ ✓ (1 GPU)   │ ✓ (1 GPU)   │                             │
│  │ A3B (FP16)    │ ✗           │ ✓ (1 GPU)   │                             │
│  │ 8B (FP16)     │ ✓ (1 GPU)   │ ✓ (1 GPU)   │                             │
│  │ 32B (FP16)    │ ✓ (2 GPUs)  │ ✓ (1 GPU)   │                             │
│  │ A30B (FP16)   │ ✗           │ ✓ (8 GPUs)  │                             │
│  │ 72B (FP16)    │ ✓ (4 GPUs)  │ ✓ (2 GPUs)  │                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### H100 (Hopper)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA H100                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VARIANTS:                                                                  │
│  - H100 PCIe: 80GB HBM3, 2 TB/s, air-cooled                               │
│  - H100 SXM5: 80GB HBM3, 3.35 TB/s, NVLink (RECOMMENDED)                  │
│  - H100 NVL: 94GB HBM3, liquid-cooled, NVLink                             │
│                                                                             │
│  KEY INNOVATIONS:                                                           │
│  ✓ FP8 support (2x throughput vs FP16)                                     │
│  ✓ Transformer Engine (automatic mixed precision)                          │
│  ✓ 3x memory bandwidth vs A100                                             │
│  ✓ 3x FP16 performance vs A100                                             │
│                                                                             │
│  BEST FOR:                                                                  │
│  ✓ Fastest training on current hardware                                    │
│  ✓ FP8 inference with minimal quality loss                                 │
│  ✓ Large batch training                                                    │
│                                                                             │
│  LIMITATIONS:                                                               │
│  ✗ High cost (~$30K+ per GPU)                                              │
│  ✗ High power consumption (700W)                                           │
│  ✗ Limited availability                                                    │
│                                                                             │
│  QWEN3-VL CAPACITY:                                                         │
│  │ Model         │ H100-80GB   │ H100 NVL    │                             │
│  ├───────────────┼─────────────┼─────────────┤                             │
│  │ 3B (FP16)     │ ✓ (1 GPU)   │ ✓ (1 GPU)   │                             │
│  │ A3B (FP16)    │ ✓ (1 GPU)   │ ✓ (1 GPU)   │                             │
│  │ 8B (FP16)     │ ✓ (1 GPU)   │ ✓ (1 GPU)   │                             │
│  │ 32B (FP16)    │ ✓ (1 GPU)   │ ✓ (1 GPU)   │                             │
│  │ A30B (FP16)   │ ✓ (8 GPUs)  │ ✓ (6 GPUs)  │                             │
│  │ 72B (FP16)    │ ✓ (2 GPUs)  │ ✓ (2 GPUs)  │                             │
│                                                                             │
│  FP8 BENEFIT:                                                               │
│  With FP8 quantization, effective capacity doubles:                        │
│  - 32B fits on 1 H100 with FP8                                             │
│  - 72B fits on 1 H100 with FP8                                             │
│  - Throughput increases 1.5-2x                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B200 (Blackwell)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA B200                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ARCHITECTURE:                                                              │
│  - Two Blackwell dies connected via 10 TB/s chip-to-chip link             │
│  - Acts as single unified GPU                                              │
│  - 192GB HBM3e unified memory                                              │
│                                                                             │
│  KEY INNOVATIONS:                                                           │
│  ✓ 2.5x FP8 performance vs H100                                            │
│  ✓ 4x memory (192GB vs 80GB)                                               │
│  ✓ 2.4x memory bandwidth (8 TB/s)                                          │
│  ✓ Second-gen Transformer Engine                                           │
│  ✓ Native support for very long context (up to 128K)                       │
│                                                                             │
│  BEST FOR:                                                                  │
│  ✓ Training largest models (72B+ on single GPU)                            │
│  ✓ Very long context inference                                             │
│  ✓ Highest throughput inference                                            │
│                                                                             │
│  LIMITATIONS:                                                               │
│  ✗ Very high power (1000W)                                                 │
│  ✗ Requires liquid cooling                                                 │
│  ✗ Limited availability (2024-2025)                                        │
│  ✗ Very expensive (~$40-50K)                                               │
│                                                                             │
│  QWEN3-VL CAPACITY:                                                         │
│  │ Model         │ B200-192GB  │ Notes                    │                │
│  ├───────────────┼─────────────┼──────────────────────────┤                │
│  │ 3B (FP16)     │ ✓ (1 GPU)   │ Overkill                 │                │
│  │ A3B (FP16)    │ ✓ (1 GPU)   │ Easy fit                 │                │
│  │ 8B (FP16)     │ ✓ (1 GPU)   │ Easy fit                 │                │
│  │ 32B (FP16)    │ ✓ (1 GPU)   │ Easy fit                 │                │
│  │ A30B (FP16)   │ ✓ (4 GPUs)  │ Half of H100 requirement │                │
│  │ 72B (FP16)    │ ✓ (1 GPU)   │ First GPU to fit 72B!   │                │
│  │ 72B (FP8)     │ ✓ (1 GPU)   │ With room for KV cache  │                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GB200 (Grace-Blackwell Superchip)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA GB200                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ARCHITECTURE:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      GB200 SUPERCHIP                                │   │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │   │
│  │  │   B200 GPU  │◄───►│ NVLink-C2C  │◄───►│  Grace CPU  │           │   │
│  │  │   192GB     │     │  900 GB/s   │     │  72 cores   │           │   │
│  │  └─────────────┘     └─────────────┘     └─────────────┘           │   │
│  │                                                │                    │   │
│  │                                           ┌────┴────┐               │   │
│  │                                           │ LPDDR5X │               │   │
│  │                                           │  512GB  │               │   │
│  │                                           └─────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TOTAL MEMORY: 192GB HBM3e + 512GB LPDDR5X = 704GB unified                │
│                                                                             │
│  KEY INNOVATIONS:                                                           │
│  ✓ CPU + GPU unified memory                                                │
│  ✓ Eliminates PCIe bottleneck                                             │
│  ✓ Ideal for inference (CPU handles preprocessing)                        │
│  ✓ NVL72: 72 GPUs in single rack with NVLink                              │
│                                                                             │
│  NVL72 SYSTEM:                                                              │
│  - 36 Grace CPUs + 72 Blackwell GPUs                                       │
│  - 13.5 TB unified HBM3e memory                                            │
│  - Acts as single 13.5TB GPU                                               │
│  - 30x faster inference than H100 DGX for LLMs                            │
│                                                                             │
│  QWEN3-VL ON GB200 NVL72:                                                  │
│  │ Model         │ GPUs Needed │ Notes                    │                │
│  ├───────────────┼─────────────┼──────────────────────────┤                │
│  │ A30B (FP16)   │ 4           │ Trivial                  │                │
│  │ 72B (FP16)    │ 1           │ Single GPU!              │                │
│  │ 405B (FP16)   │ 4           │ Llama 3.1 class          │                │
│  │ 1T+ (FP8)     │ 72          │ Entire rack              │                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GB300 (Blackwell Ultra)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA GB300 (Blackwell Ultra)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STATUS: Announced 2025, shipping late 2025                                │
│                                                                             │
│  ARCHITECTURE:                                                              │
│  - Enhanced Blackwell "Ultra" GPU                                          │
│  - 288GB HBM3e (50% more than B200/GB200)                                 │
│  - Same 8 TB/s bandwidth                                                   │
│  - ~20 PFLOPS AI compute                                                   │
│                                                                             │
│  KEY IMPROVEMENTS OVER GB200:                                               │
│  ✓ 50% more memory (288GB vs 192GB)                                        │
│  ✓ ~20% more compute                                                       │
│  ✓ Better power efficiency                                                 │
│  ✓ Enhanced NVLink (130 TB/s for NVL72)                                   │
│                                                                             │
│  GB300 NVL72:                                                               │
│  - 20.7 TB unified memory (vs 13.5 TB)                                     │
│  - Purpose-built for test-time scaling (reasoning)                        │
│  - Optimized for AI agents and multi-step inference                       │
│                                                                             │
│  BEST FOR:                                                                  │
│  ✓ Largest models (1T+ parameters)                                         │
│  ✓ Very long context (128K+ tokens)                                        │
│  ✓ Agent workloads with many inference steps                              │
│  ✓ Future-proofing                                                         │
│                                                                             │
│  PRICING: Expected $80-100K+ per superchip                                 │
│  AVAILABILITY: Late 2025, very limited                                     │
│                                                                             │
│  NOTE: GB300 is NOT "Rubin" architecture.                                  │
│  Rubin is next-gen (2026+), GB300 is enhanced Blackwell.                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.3 PERFORMANCE COMPARISON

### Training Performance (Tokens/Second)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING THROUGHPUT COMPARISON                           │
│                    (Qwen3-VL-32B, batch size 1, FP16)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  A100-80GB:  ████████████████                           1,200 tok/s        │
│  H100-80GB:  ████████████████████████████████           3,400 tok/s        │
│  B200-192GB: ████████████████████████████████████████   5,800 tok/s        │
│  GB200:      ██████████████████████████████████████████ 6,200 tok/s        │
│                                                                             │
│  Relative speedup vs A100:                                                  │
│  - H100: 2.8x faster                                                       │
│  - B200: 4.8x faster                                                       │
│  - GB200: 5.2x faster                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Inference Performance (Tokens/Second)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE THROUGHPUT (Generation)                        │
│                    (Qwen3-VL-A3B, batch size 32, per GPU)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         │ FP16        │ FP8         │ AWQ 4-bit   │        │
│  ├──────────────────────┼─────────────┼─────────────┼─────────────┤        │
│  │ A100-80GB            │ 1,800       │ N/A         │ 2,400       │        │
│  │ H100-80GB            │ 4,200       │ 7,500       │ 5,600       │        │
│  │ B200-192GB           │ 7,800       │ 14,000      │ 10,500      │        │
│  │ GB200 (superchip)    │ 8,500       │ 15,200      │ 11,400      │        │
│                                                                             │
│  NOTE: FP8 gives ~1.8x speedup on Hopper+ GPUs                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.4 COST-PERFORMANCE ANALYSIS

### Cloud Pricing (Approximate, 2025)

| GPU | On-Demand/hr | Spot/hr | Monthly Reserved |
|-----|--------------|---------|------------------|
| A100-40GB | $3.50 | $1.20 | $2,500 |
| A100-80GB | $5.00 | $1.80 | $3,600 |
| H100-80GB | $8.00 | $3.00 | $5,800 |
| H100 SXM | $10.00 | $4.00 | $7,200 |
| B200 | TBD | TBD | TBD |

### Cost Per Training Run

| Model | GPU Config | Time | Cost (Spot) | Cost (On-Demand) |
|-------|------------|------|-------------|------------------|
| **3B SFT** | 1×A100 | 2hr | $2.40 | $7.00 |
| **A3B SFT** | 1×A100 | 4hr | $7.20 | $20.00 |
| **32B SFT** | 2×H100 | 6hr | $36.00 | $96.00 |
| **A30B SFT** | 8×H100 | 12hr | $288.00 | $768.00 |
| **Embedding** | 1×A100 | 3hr | $5.40 | $15.00 |
| **Reranker** | 1×A100 | 1.5hr | $2.70 | $7.50 |
| **DPO** | 1×A100 | 2hr | $3.60 | $10.00 |

### Cost-Efficiency Recommendations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COST OPTIMIZATION GUIDE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FOR TRAINING:                                                              │
│  ────────────                                                               │
│  1. Use SPOT instances (60-70% savings)                                    │
│  2. Use A100s for models ≤32B (best $/performance)                         │
│  3. Use H100s for models >32B or when time is critical                     │
│  4. Use QLoRA to reduce GPU requirements                                   │
│                                                                             │
│  FOR INFERENCE:                                                             │
│  ──────────────                                                             │
│  1. Use AWQ/GPTQ quantization (4x memory reduction)                        │
│  2. Use FP8 on H100/B200 (2x speedup, minimal quality loss)               │
│  3. Batch requests for throughput                                          │
│  4. Use reserved instances for predictable workloads                       │
│                                                                             │
│  HIGGSFIELD RECOMMENDED SETUP:                                              │
│  ─────────────────────────────                                              │
│                                                                             │
│  Development:                                                               │
│  - 1× A100-80GB (cloud spot): ~$2/hr                                       │
│  - Run: Embedding + Reranker + A3B generation                              │
│                                                                             │
│  Production (low traffic):                                                  │
│  - 1× A100-80GB reserved: ~$120/day                                        │
│  - Quantized A3B with AWQ                                                  │
│  - ~20 requests/second capacity                                            │
│                                                                             │
│  Production (high traffic):                                                 │
│  - 2× H100-80GB reserved: ~$400/day                                        │
│  - FP8 quantization                                                        │
│  - ~100 requests/second capacity                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.5 CONFIGURATION BY GPU

### A100 Configuration

```bash
# Optimal A100 configuration for Qwen3-VL-A3B

# Training
torchrun --nproc_per_node=1 train.py \
    --model Qwen/Qwen3-VL-A3B-Instruct \
    --batch_size 2 \
    --gradient_accumulation 16 \
    --bf16 \
    --gradient_checkpointing \
    --flash_attn

# Inference (vLLM)
vllm serve Qwen/Qwen3-VL-A3B-Instruct \
    --tensor-parallel-size 1 \
    --quantization awq \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 64 \
    --enable-prefix-caching
```

### H100 Configuration

```bash
# Optimal H100 configuration for Qwen3-VL-A3B

# Training (leverage FP8)
torchrun --nproc_per_node=1 train.py \
    --model Qwen/Qwen3-VL-A3B-Instruct \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --bf16 \
    --use_flash_attn_3 \
    --transformer_engine  # Auto FP8

# Inference (FP8)
vllm serve Qwen/Qwen3-VL-A3B-Instruct \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 128 \
    --enable-prefix-caching
```

### B200/GB200 Configuration

```bash
# Optimal B200/GB200 configuration

# Can fit much larger models or bigger batches
vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --tensor-parallel-size 1 \  # 32B fits on single B200!
    --quantization fp8 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --max-model-len 32768  # Long context

# For A30B on GB200
vllm serve Qwen/Qwen3-VL-A30B-Instruct \
    --tensor-parallel-size 2 \
    --quantization fp8 \
    --enable-expert-parallel
```

---

## 8.6 MEMORY CALCULATOR

```python
# GPU MEMORY CALCULATOR

def estimate_memory(
    model_params_billions: float,
    precision: str = "fp16",
    batch_size: int = 1,
    sequence_length: int = 2048,
    num_layers: int = 32,
    hidden_size: int = 4096,
    num_heads: int = 32,
) -> dict:
    """
    Estimate GPU memory requirements.

    Returns memory breakdown in GB.
    """

    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1,
        "int8": 1,
        "int4": 0.5,
    }

    # Model weights
    weight_memory = model_params_billions * bytes_per_param[precision]

    # KV cache (per layer, per batch)
    kv_size_per_token = 2 * hidden_size * bytes_per_param[precision] / 1e9
    kv_memory = batch_size * sequence_length * num_layers * kv_size_per_token

    # Activations (rough estimate)
    activation_memory = batch_size * sequence_length * hidden_size * num_layers * 2 / 1e9

    # Optimizer states (for training)
    # Adam: 2x model size (momentum + variance)
    optimizer_memory = 2 * weight_memory if precision in ["fp32", "fp16", "bf16"] else 0

    # Gradients
    gradient_memory = weight_memory

    return {
        "weights_gb": weight_memory,
        "kv_cache_gb": kv_memory,
        "activations_gb": activation_memory,
        "optimizer_gb": optimizer_memory,
        "gradients_gb": gradient_memory,
        "total_inference_gb": weight_memory + kv_memory + activation_memory,
        "total_training_gb": weight_memory + kv_memory + activation_memory + optimizer_memory + gradient_memory,
    }

# Examples
print("Qwen3-VL-A3B FP16 Inference:")
print(estimate_memory(30, "fp16", batch_size=1, sequence_length=2048))

print("\nQwen3-VL-A3B AWQ Inference:")
print(estimate_memory(30, "int4", batch_size=1, sequence_length=2048))

print("\nQwen3-VL-32B FP16 Training:")
print(estimate_memory(32, "fp16", batch_size=1, sequence_length=2048))
```

---

## NEXT: Part 9 - Complete SDLC/MLOps Pipeline

Continue to [PART9_MLOPS.md](./PART9_MLOPS.md)
