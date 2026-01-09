#!/usr/bin/env python3
"""
MINIMAL Reranker Model Fine-Tuning for Higgsfield Automation
=============================================================

Simplest possible reranker fine-tuning script.
Binary classification: relevant (1) vs not relevant (0).

Requirements:
    pip install transformers peft bitsandbytes accelerate torch pillow

Run:
    python train_reranker.py

GPU Memory: ~20GB
Time: ~1-2 hours for 300 examples, 3 epochs
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "Alibaba-NLP/Qwen3-VL-Reranker-8B"  # or 2B
DATA_PATH = "data/reranker_train.jsonl"
OUTPUT_DIR = "outputs/reranker_higgsfield"
SCREENSHOT_DIR = "data/screenshots"

EPOCHS = 3
BATCH_SIZE = 1  # Reranker uses longer sequences
LEARNING_RATE = 1e-4
LORA_R = 32
LORA_ALPHA = 64
MAX_LENGTH = 2048


# =============================================================================
# MODEL
# =============================================================================

class RerankerWithLoss(nn.Module):
    """Reranker using yes/no logit difference."""

    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer

        # Get yes/no token IDs
        self.yes_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_id = tokenizer.convert_tokens_to_ids("no")

        # Binary head from LM weights
        self.binary_head = self._create_binary_head()

    def _create_binary_head(self):
        """Create (W_yes - W_no) projection."""
        w_yes = self.model.lm_head.weight[self.yes_id]
        w_no = self.model.lm_head.weight[self.no_id]

        head = nn.Linear(w_yes.size(0), 1, bias=False)
        with torch.no_grad():
            head.weight[0] = w_yes - w_no
        return head

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # Get hidden states (not logits)
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k not in ['labels']}
        )

        # Get last token hidden state
        hidden = outputs.last_hidden_state
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, seq_lens]

        # Binary logits
        logits = self.binary_head(last_hidden).squeeze(-1)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return {"loss": loss, "logits": logits, "scores": torch.sigmoid(logits)}


# =============================================================================
# DATASET
# =============================================================================

RERANKER_PROMPT = """Judge whether the Document meets the requirements based on the Query. Answer only 'yes' or 'no'.

<Query>: {query_text}
<Document>: {doc_text}"""


class RerankerDataset(Dataset):
    def __init__(self, data_path, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length

        with open(data_path) as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(self.data)} reranker examples")
        pos = sum(1 for d in self.data if d.get("label", 0) == 1)
        print(f"  Positive: {pos}, Negative: {len(self.data) - pos}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "document": item["document"],
            "label": item.get("label", 0)
        }


def collate_fn(batch, processor, max_length):
    """Format as reranker input."""
    texts = []
    images = []
    labels = []

    for item in batch:
        query_text = item["query"].get("text", "")
        doc_text = item["document"].get("text", "")

        prompt = RERANKER_PROMPT.format(query_text=query_text, doc_text=doc_text)
        texts.append(prompt)

        # Load images
        for key in ["query", "document"]:
            img_path = item[key].get("image")
            if img_path:
                full_path = Path(SCREENSHOT_DIR) / img_path
                if full_path.exists():
                    images.append(Image.open(full_path).convert("RGB"))

        labels.append(item["label"])

    inputs = processor(
        text=texts,
        images=images if images else None,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    inputs["labels"] = torch.tensor(labels, dtype=torch.float)
    return inputs


# =============================================================================
# TRAINING
# =============================================================================

def main():
    print("=" * 60)
    print("RERANKER MODEL FINE-TUNING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model
    print(f"Loading model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for LoRA
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Wrap in reranker
    model = RerankerWithLoss(base_model, processor.tokenizer)

    # Dataset
    dataset = RerankerDataset(DATA_PATH, processor, MAX_LENGTH)

    def collate(batch):
        return collate_fn(batch, processor, MAX_LENGTH)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training
    print(f"\nStarting training: {EPOCHS} epochs")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy
            preds = (outputs["scores"] > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.2%}"})

        print(f"Epoch {epoch+1} - Loss: {total_loss/len(dataloader):.4f}, Acc: {correct/total:.2%}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
