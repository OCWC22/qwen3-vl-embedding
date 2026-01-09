#!/usr/bin/env python3
"""
MINIMAL Embedding Model Fine-Tuning for Higgsfield Automation
==============================================================

This is the SIMPLEST possible script to fine-tune Qwen3-VL-Embedding.
No bells and whistles - just the essentials.

Requirements:
    pip install transformers peft bitsandbytes accelerate torch pillow

Run:
    python train_embedding.py

GPU Memory: ~20GB (fits on RTX 3090/4090/A100)
Time: ~2-4 hours for 500 examples, 3 epochs
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLPreTrainedModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =============================================================================
# CONFIG - CHANGE THESE
# =============================================================================

MODEL_NAME = "Alibaba-NLP/Qwen3-VL-Embedding-8B"  # or 2B for faster training
DATA_PATH = "data/embedding_train.jsonl"
OUTPUT_DIR = "outputs/embedding_higgsfield"
SCREENSHOT_DIR = "data/screenshots"

# Training params
EPOCHS = 3
BATCH_SIZE = 2  # Increase if you have more VRAM
LEARNING_RATE = 2e-4
LORA_R = 32  # Lower = faster, higher = more capacity
LORA_ALPHA = 64
MAX_LENGTH = 1024  # Reduce for faster training


# =============================================================================
# MODEL
# =============================================================================

class Qwen3VLEmbedding(Qwen3VLPreTrainedModel):
    """Minimal embedding model - just the encoder + EOS pooling."""

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.temperature = 0.02

    def pool_eos(self, hidden_states, attention_mask):
        """Extract last token (EOS) hidden state."""
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lens]

    def encode(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        """Get normalized embeddings."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        emb = self.pool_eos(outputs.last_hidden_state, attention_mask)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, query, positive, negative=None):
        """Compute contrastive loss."""
        q_emb = self.encode(**query)
        p_emb = self.encode(**positive)

        # Positive similarity
        pos_sim = (q_emb * p_emb).sum(dim=-1) / self.temperature

        # In-batch negatives
        all_sim = torch.mm(q_emb, p_emb.t()) / self.temperature

        # Add explicit negatives if provided
        if negative is not None:
            n_emb = self.encode(**negative)
            neg_sim = torch.mm(q_emb, n_emb.t()) / self.temperature
            all_sim = torch.cat([all_sim, neg_sim], dim=1)

        # InfoNCE loss
        labels = torch.arange(q_emb.size(0), device=q_emb.device)
        loss = F.cross_entropy(all_sim, labels)

        return {"loss": loss}


# =============================================================================
# DATASET
# =============================================================================

class EmbeddingDataset(Dataset):
    def __init__(self, data_path, processor, max_length=1024):
        self.processor = processor
        self.max_length = max_length

        with open(data_path) as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def load_image(self, path):
        full_path = Path(SCREENSHOT_DIR) / path if not Path(path).is_absolute() else path
        if full_path.exists():
            return Image.open(full_path).convert("RGB")
        # Return placeholder if image doesn't exist
        return Image.new("RGB", (224, 224), color="gray")

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": item.get("negatives", [{}])[0] if item.get("negatives") else None
        }


def collate_fn(batch, processor, max_length):
    """Collate batch into model inputs."""

    def process_items(items):
        texts = []
        images = []

        for item in items:
            if item is None:
                continue
            text = item.get("text", "")
            texts.append(f"Represent: {text}")

            if item.get("image"):
                img_path = Path(SCREENSHOT_DIR) / item["image"]
                if img_path.exists():
                    images.append(Image.open(img_path).convert("RGB"))

        if not texts:
            return None

        inputs = processor(
            text=texts,
            images=images if images else None,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return inputs

    queries = [b["query"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = [b["negative"] for b in batch if b["negative"]]

    return {
        "query": process_items(queries),
        "positive": process_items(positives),
        "negative": process_items(negatives) if negatives else None
    }


# =============================================================================
# TRAINING
# =============================================================================

def main():
    print("=" * 60)
    print("EMBEDDING MODEL FINE-TUNING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load processor
    print(f"Loading processor: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = Qwen3VLEmbedding.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = EmbeddingDataset(DATA_PATH, processor, MAX_LENGTH)

    def collate(batch):
        return collate_fn(batch, processor, MAX_LENGTH)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nStarting training: {EPOCHS} epochs, {len(dataloader)} batches/epoch")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            if batch["query"] is None:
                continue

            # Move to device
            query = {k: v.to(device) for k, v in batch["query"].items()}
            positive = {k: v.to(device) for k, v in batch["positive"].items()}
            negative = None
            if batch["negative"] is not None:
                negative = {k: v.to(device) for k, v in batch["negative"].items()}

            # Forward
            outputs = model(query, positive, negative)
            loss = outputs["loss"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
