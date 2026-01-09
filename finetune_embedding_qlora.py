"""
QLoRA Fine-Tuning for Qwen3-VL-Embedding Model
===============================================

This script fine-tunes the Qwen3-VL-Embedding model using QLoRA (4-bit quantization + LoRA)
for domain-specific embedding tasks like computer use agents.

Training Objective: Contrastive Learning with InfoNCE Loss
- Pull positive pairs close in embedding space
- Push negative pairs apart

Usage:
    python finetune_embedding_qlora.py \
        --model_name Alibaba-NLP/Qwen3-VL-Embedding-8B \
        --train_data data/embedding_train.jsonl \
        --output_dir outputs/embedding_finetuned \
        --num_epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import numpy as np

from transformers import (
    Qwen3VLProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils.vision_process import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class Qwen3VLForEmbeddingWithLoss(Qwen3VLPreTrainedModel):
    """
    Qwen3-VL model for embedding with contrastive loss.
    Removes the LM head and uses EOS token pooling for embeddings.
    """

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.temperature = 0.02  # InfoNCE temperature
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract the last non-padding token's hidden state (EOS pooling)."""
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Encode inputs to normalized embeddings."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs
        )
        embeddings = self._pooling_last(outputs.last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def forward(
        self,
        query_input_ids: torch.LongTensor = None,
        query_attention_mask: torch.Tensor = None,
        query_pixel_values: Optional[torch.Tensor] = None,
        query_image_grid_thw: Optional[torch.LongTensor] = None,
        pos_input_ids: torch.LongTensor = None,
        pos_attention_mask: torch.Tensor = None,
        pos_pixel_values: Optional[torch.Tensor] = None,
        pos_image_grid_thw: Optional[torch.LongTensor] = None,
        neg_input_ids: Optional[torch.LongTensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
        neg_pixel_values: Optional[torch.Tensor] = None,
        neg_image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Forward pass with contrastive loss computation.

        Args:
            query_*: Query inputs (screenshot + action description)
            pos_*: Positive document inputs (matching example)
            neg_*: Negative document inputs (non-matching examples, optional)

        Returns:
            dict with 'loss' and 'embeddings'
        """
        # Encode query
        query_emb = self.encode(
            query_input_ids, query_attention_mask,
            query_pixel_values, query_image_grid_thw
        )

        # Encode positive
        pos_emb = self.encode(
            pos_input_ids, pos_attention_mask,
            pos_pixel_values, pos_image_grid_thw
        )

        # Encode negatives if provided
        neg_emb = None
        if neg_input_ids is not None:
            neg_emb = self.encode(
                neg_input_ids, neg_attention_mask,
                neg_pixel_values, neg_image_grid_thw
            )

        # Compute InfoNCE loss
        loss = self.compute_contrastive_loss(query_emb, pos_emb, neg_emb)

        return {
            'loss': loss,
            'query_embeddings': query_emb,
            'pos_embeddings': pos_emb,
            'neg_embeddings': neg_emb,
        }

    def compute_contrastive_loss(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Uses in-batch negatives + explicit negatives if provided.
        """
        batch_size = query_emb.size(0)

        # Similarity with positive: [batch_size]
        pos_sim = torch.sum(query_emb * pos_emb, dim=-1) / self.temperature

        # In-batch negatives: all other positives become negatives
        # [batch_size, batch_size]
        all_sim = torch.mm(query_emb, pos_emb.t()) / self.temperature

        # Add explicit negatives if provided
        if neg_emb is not None:
            # [batch_size, num_neg]
            neg_sim = torch.mm(query_emb, neg_emb.t()) / self.temperature
            all_sim = torch.cat([all_sim, neg_sim], dim=1)

        # Labels: positive is always at diagonal (index i for query i)
        labels = torch.arange(batch_size, device=query_emb.device)

        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(all_sim, labels)

        return loss


# ============================================================================
# DATASET
# ============================================================================

class EmbeddingDataset(Dataset):
    """
    Dataset for contrastive embedding training.

    Expected JSONL format:
    {
        "query": {"image": "path/to/img.png", "text": "click the login button"},
        "positive": {"image": "path/to/pos.png", "text": "login button action"},
        "negatives": [
            {"image": "path/to/neg1.png", "text": "cancel button"},
            {"image": "path/to/neg2.png", "text": "submit in different form"}
        ]
    }
    """

    def __init__(
        self,
        data_path: str,
        processor: Qwen3VLProcessor,
        max_length: int = 2048,
        image_base_path: str = "",
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_base_path = image_base_path

        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} training examples")

    def __len__(self):
        return len(self.data)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path."""
        if self.image_base_path:
            image_path = os.path.join(self.image_base_path, image_path)
        return Image.open(image_path).convert('RGB')

    def _format_input(self, item: Dict) -> List[Dict]:
        """Format item as Qwen3-VL conversation."""
        content = []

        if item.get('image'):
            content.append({
                'type': 'image',
                'image': self._load_image(item['image'])
            })

        if item.get('text'):
            content.append({
                'type': 'text',
                'text': item['text']
            })

        return [
            {"role": "system", "content": [{"type": "text", "text": "Represent the visual content and action."}]},
            {"role": "user", "content": content}
        ]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Format query, positive, and one random negative
        query_conv = self._format_input(item['query'])
        pos_conv = self._format_input(item['positive'])

        # Select one random negative (or None if not available)
        neg_conv = None
        if item.get('negatives') and len(item['negatives']) > 0:
            neg_item = np.random.choice(item['negatives'])
            neg_conv = self._format_input(neg_item)

        return {
            'query_conv': query_conv,
            'pos_conv': pos_conv,
            'neg_conv': neg_conv,
        }


class EmbeddingCollator:
    """
    Collator for batching embedding training data.
    Processes conversations into model inputs.
    """

    def __init__(self, processor: Qwen3VLProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def _process_conversations(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Process a batch of conversations into model inputs."""
        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        # Extract images from conversations
        all_images = []
        for conv in conversations:
            images = []
            for msg in conv:
                for content in msg.get('content', []):
                    if content.get('type') == 'image' and isinstance(content.get('image'), Image.Image):
                        images.append(content['image'])
            all_images.append(images if images else None)

        # Flatten images for processor
        flat_images = []
        for imgs in all_images:
            if imgs:
                flat_images.extend(imgs)

        inputs = self.processor(
            text=texts,
            images=flat_images if flat_images else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return inputs

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        query_convs = [item['query_conv'] for item in batch]
        pos_convs = [item['pos_conv'] for item in batch]
        neg_convs = [item['neg_conv'] for item in batch if item['neg_conv'] is not None]

        # Process queries
        query_inputs = self._process_conversations(query_convs)

        # Process positives
        pos_inputs = self._process_conversations(pos_convs)

        # Process negatives (if any)
        neg_inputs = None
        if neg_convs:
            neg_inputs = self._process_conversations(neg_convs)

        # Rename keys with prefixes
        result = {}
        for key, value in query_inputs.items():
            result[f'query_{key}'] = value
        for key, value in pos_inputs.items():
            result[f'pos_{key}'] = value
        if neg_inputs:
            for key, value in neg_inputs.items():
                result[f'neg_{key}'] = value

        return result


# ============================================================================
# TRAINING
# ============================================================================

def setup_qlora_model(model_name: str, lora_r: int = 64, lora_alpha: int = 128):
    """
    Setup model with QLoRA configuration.

    Args:
        model_name: HuggingFace model name or path
        lora_r: LoRA rank (higher = more capacity, more memory)
        lora_alpha: LoRA alpha (scaling factor)

    Returns:
        model, processor tuple
    """
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with quantization
    logger.info(f"Loading model: {model_name}")
    model = Qwen3VLForEmbeddingWithLoss.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    # Target modules for Qwen3-VL (attention + MLP)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load processor
    processor = Qwen3VLProcessor.from_pretrained(model_name, padding_side='right')

    return model, processor


def train(
    model_name: str,
    train_data: str,
    output_dir: str,
    val_data: Optional[str] = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    max_length: int = 2048,
    lora_r: int = 64,
    lora_alpha: int = 128,
    image_base_path: str = "",
    save_steps: int = 500,
    logging_steps: int = 10,
):
    """
    Main training function.
    """
    # Setup model
    model, processor = setup_qlora_model(model_name, lora_r, lora_alpha)

    # Create datasets
    train_dataset = EmbeddingDataset(
        train_data, processor, max_length, image_base_path
    )

    val_dataset = None
    if val_data:
        val_dataset = EmbeddingDataset(
            val_data, processor, max_length, image_base_path
        )

    # Create collator
    collator = EmbeddingCollator(processor, max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    logger.info("Training complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3-VL-Embedding")

    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/Qwen3-VL-Embedding-8B",
                        help="Base model name or path")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="outputs/embedding_finetuned",
                        help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--image_base_path", type=str, default="",
                        help="Base path for images")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        train_data=args.train_data,
        output_dir=args.output_dir,
        val_data=args.val_data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        image_base_path=args.image_base_path,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
