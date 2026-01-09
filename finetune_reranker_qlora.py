"""
QLoRA Fine-Tuning for Qwen3-VL-Reranker Model
==============================================

This script fine-tunes the Qwen3-VL-Reranker model using QLoRA for
domain-specific reranking tasks like computer use agents.

Training Objective: Binary Classification (relevant vs not relevant)
- Uses yes/no token logit difference for scoring
- Binary cross-entropy loss

Usage:
    python finetune_reranker_qlora.py \
        --model_name Alibaba-NLP/Qwen3-VL-Reranker-8B \
        --train_data data/reranker_train.jsonl \
        --output_dir outputs/reranker_finetuned \
        --num_epochs 3 \
        --batch_size 2 \
        --learning_rate 1e-4
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import numpy as np

from transformers import (
    Qwen3VLProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

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

class Qwen3VLRerankerWithLoss(nn.Module):
    """
    Qwen3-VL Reranker model with binary classification loss.

    Uses the yes/no token logit difference trick:
    score = sigmoid((W_yes - W_no) · hidden_state)

    This is more efficient than generating yes/no tokens.
    """

    def __init__(self, base_model: Qwen3VLForConditionalGeneration, tokenizer):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer

        # Get yes/no token IDs
        self.token_yes = tokenizer.convert_tokens_to_ids("yes")
        self.token_no = tokenizer.convert_tokens_to_ids("no")

        # Create binary classification head from LM head weights
        self.binary_head = self._create_binary_head()

    def _create_binary_head(self) -> nn.Linear:
        """
        Create a linear layer that computes: (W_yes - W_no) · hidden

        This is equivalent to computing:
        logit_yes - logit_no = W_yes @ h - W_no @ h = (W_yes - W_no) @ h
        """
        lm_head_weights = self.model.lm_head.weight.data
        weight_yes = lm_head_weights[self.token_yes]  # [hidden_dim]
        weight_no = lm_head_weights[self.token_no]    # [hidden_dim]

        hidden_dim = weight_yes.size(0)
        binary_head = nn.Linear(hidden_dim, 1, bias=False)

        with torch.no_grad():
            binary_head.weight[0] = weight_yes - weight_no

        return binary_head

    @staticmethod
    def _get_last_hidden_state(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract the last non-padding token's hidden state."""
        # Find the last non-zero position in attention mask
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, seq_lengths]
        return last_hidden

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,  # Binary labels: 0 or 1
        **kwargs
    ):
        """
        Forward pass with binary classification.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            pixel_values: Image pixel values
            image_grid_thw: Image grid info
            labels: Binary labels [batch_size] where 1=relevant, 0=not relevant

        Returns:
            dict with 'loss', 'logits', 'scores'
        """
        # Forward through base model (without LM head)
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            **kwargs
        )

        # Get last layer hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Extract last token hidden state
        last_hidden = self._get_last_hidden_state(hidden_states, attention_mask)

        # Compute binary logits: (W_yes - W_no) · h
        logits = self.binary_head(last_hidden).squeeze(-1)  # [batch]

        # Compute scores (probabilities)
        scores = torch.sigmoid(logits)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            labels = labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'scores': scores,
        }


# ============================================================================
# DATASET
# ============================================================================

class RerankerDataset(Dataset):
    """
    Dataset for reranker binary classification training.

    Expected JSONL format:
    {
        "query": {"image": "path/to/query.png", "text": "click login button"},
        "document": {"image": "path/to/doc.png", "text": "login button action"},
        "label": 1  // 1 = relevant, 0 = not relevant
    }
    """

    # Reranker prompt template
    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be 'yes' or 'no'."
    )

    def __init__(
        self,
        data_path: str,
        processor: Qwen3VLProcessor,
        max_length: int = 4096,
        image_base_path: str = "",
        instruct: str = "Find the most relevant action for the given screenshot and query.",
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_base_path = image_base_path
        self.instruct = instruct

        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} reranker training examples")

        # Count label distribution
        pos_count = sum(1 for d in self.data if d.get('label', 0) == 1)
        neg_count = len(self.data) - pos_count
        logger.info(f"Label distribution: {pos_count} positive, {neg_count} negative")

    def __len__(self):
        return len(self.data)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path."""
        if self.image_base_path:
            image_path = os.path.join(self.image_base_path, image_path)
        return Image.open(image_path).convert('RGB')

    def _format_reranker_input(self, query: Dict, document: Dict) -> List[Dict]:
        """
        Format query-document pair as reranker conversation.

        Format:
        System: [Judge prompt]
        User:
          <Instruct>: [instruction]
          <Query>: [query image + text]
          <Document>: [document image + text]
        """
        user_content = []

        # Add instruct
        user_content.append({
            'type': 'text',
            'text': f'<Instruct>: {self.instruct}\n<Query>: '
        })

        # Add query image
        if query.get('image'):
            user_content.append({
                'type': 'image',
                'image': self._load_image(query['image'])
            })

        # Add query text
        if query.get('text'):
            user_content.append({
                'type': 'text',
                'text': f'{query["text"]}\n<Document>: '
            })
        else:
            user_content.append({
                'type': 'text',
                'text': '\n<Document>: '
            })

        # Add document image
        if document.get('image'):
            user_content.append({
                'type': 'image',
                'image': self._load_image(document['image'])
            })

        # Add document text
        if document.get('text'):
            user_content.append({
                'type': 'text',
                'text': document['text']
            })

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content}
        ]

        return conversation

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        conversation = self._format_reranker_input(item['query'], item['document'])
        label = item.get('label', 0)

        return {
            'conversation': conversation,
            'label': label,
        }


class RerankerCollator:
    """
    Collator for batching reranker training data.
    """

    def __init__(self, processor: Qwen3VLProcessor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        conversations = [item['conversation'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)

        # Process conversations
        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        # Extract images
        all_images = []
        for conv in conversations:
            for msg in conv:
                for content in msg.get('content', []):
                    if content.get('type') == 'image' and isinstance(content.get('image'), Image.Image):
                        all_images.append(content['image'])

        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        inputs['labels'] = labels
        return inputs


# ============================================================================
# TRAINING
# ============================================================================

def setup_reranker_qlora(model_name: str, lora_r: int = 64, lora_alpha: int = 128):
    """
    Setup reranker model with QLoRA configuration.
    """
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    logger.info(f"Loading model: {model_name}")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load processor
    processor = Qwen3VLProcessor.from_pretrained(model_name, padding_side='right')

    # Prepare for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence classification
    )

    # Apply LoRA to base model
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Wrap in reranker model
    model = Qwen3VLRerankerWithLoss(base_model, processor.tokenizer)

    return model, processor


class RerankerTrainer(Trainer):
    """Custom trainer for reranker model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss for reranker."""
        labels = inputs.pop('labels', None)
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']

        if return_outputs:
            return loss, outputs
        return loss


def train(
    model_name: str,
    train_data: str,
    output_dir: str,
    val_data: Optional[str] = None,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    max_length: int = 4096,
    lora_r: int = 64,
    lora_alpha: int = 128,
    image_base_path: str = "",
    instruct: str = "Find the most relevant action for the given screenshot and query.",
    save_steps: int = 500,
    logging_steps: int = 10,
):
    """
    Main training function for reranker.
    """
    # Setup model
    model, processor = setup_reranker_qlora(model_name, lora_r, lora_alpha)

    # Create datasets
    train_dataset = RerankerDataset(
        train_data, processor, max_length, image_base_path, instruct
    )

    val_dataset = None
    if val_data:
        val_dataset = RerankerDataset(
            val_data, processor, max_length, image_base_path, instruct
        )

    # Create collator
    collator = RerankerCollator(processor, max_length)

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
    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    logger.info("Starting reranker training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}")

    # Save the LoRA weights
    model.model.save_pretrained(output_dir)  # Save PEFT model
    processor.save_pretrained(output_dir)

    logger.info("Reranker training complete!")


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_reranker(model, eval_dataloader, device):
    """
    Evaluate reranker on binary classification metrics.

    Returns:
        dict with accuracy, precision, recall, F1, AUC-ROC
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    model.eval()
    all_labels = []
    all_scores = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')

            outputs = model(**batch)
            scores = outputs['scores'].cpu().numpy()
            preds = (scores > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)
            all_preds.extend(preds)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    auc_roc = roc_auc_score(all_labels, all_scores)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3-VL-Reranker")

    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/Qwen3-VL-Reranker-8B",
                        help="Base model name or path")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="outputs/reranker_finetuned",
                        help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--image_base_path", type=str, default="",
                        help="Base path for images")
    parser.add_argument("--instruct", type=str,
                        default="Find the most relevant action for the given screenshot and query.",
                        help="Instruction for reranking task")
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
        instruct=args.instruct,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
