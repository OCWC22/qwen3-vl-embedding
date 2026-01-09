# Minimal Fine-Tuning Guide: Higgsfield AI Automation

This guide shows the **ABSOLUTE MINIMUM** needed to fine-tune all 4 components
for a computer use agent that automates Higgsfield AI video generation.

## Target Workflow
1. Login to higgsfield.ai
2. Navigate to Create > Video
3. Select model (WAN 2.6, Kling 2.6, Minimax Hailuo)
4. Upload image
5. Configure motion presets
6. Generate video

## Minimum Data Requirements

| Component | Min Examples | Recommended | Format |
|-----------|-------------|-------------|--------|
| Embedding | 500 triplets | 5,000 | (query, pos, neg) |
| Reranker | 300 pairs | 3,000 | (query, doc, label) |
| SFT | 100 examples | 1,000 | (screenshot, action) |
| RL/DPO | 50 preferences | 500 | (chosen, rejected) |

## Quick Start

```bash
# 1. Generate sample data structure
python create_sample_data.py

# 2. Fine-tune embedding (2-4 hours on single GPU)
python train_embedding.py

# 3. Fine-tune reranker (1-2 hours)
python train_reranker.py

# 4. Fine-tune base model SFT (4-8 hours)
python train_sft.py

# 5. Optional: RL/DPO refinement (2-4 hours)
python train_dpo.py
```
