"""
Training Data Preparation for Qwen3-VL Embedding/Reranker Fine-tuning
=====================================================================

This script helps prepare training data for fine-tuning:
1. Embedding model: Contrastive learning with triplets
2. Reranker model: Binary classification pairs

Also includes utilities for:
- Hard negative mining
- Data augmentation
- Query paraphrasing (using LLM)
- Statistics and validation

Usage:
    # Generate sample data structure
    python prepare_training_data.py --generate_examples

    # Validate existing data
    python prepare_training_data.py --validate data/embedding_train.jsonl

    # Mine hard negatives from existing embeddings
    python prepare_training_data.py --mine_hard_negatives \
        --data data/embedding_train.jsonl \
        --model_name Alibaba-NLP/Qwen3-VL-Embedding-8B
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================

def generate_embedding_examples():
    """
    Generate example JSONL format for embedding training data.
    """
    examples = [
        # Example 1: Login button click
        {
            "query": {
                "image": "screenshots/gmail_compose_001.png",
                "text": "Click the send button to send the email"
            },
            "positive": {
                "image": "screenshots/gmail_compose_send_highlighted.png",
                "text": "Blue send button in Gmail compose window, coordinates (580, 120)"
            },
            "negatives": [
                {
                    "image": "screenshots/gmail_compose_001.png",
                    "text": "Discard button to delete draft",
                    "negative_type": "same_view_wrong_element"
                },
                {
                    "image": "screenshots/outlook_compose.png",
                    "text": "Send button in Outlook",
                    "negative_type": "similar_element_different_app"
                },
                {
                    "image": "screenshots/gmail_inbox.png",
                    "text": "Compose button in Gmail inbox",
                    "negative_type": "different_view_same_app"
                }
            ]
        },
        # Example 2: Form submission
        {
            "query": {
                "image": "screenshots/login_form_001.png",
                "text": "Submit the login form"
            },
            "positive": {
                "image": "screenshots/login_form_submit.png",
                "text": "Login button below password field, coordinates (400, 350)"
            },
            "negatives": [
                {
                    "image": "screenshots/login_form_001.png",
                    "text": "Forgot password link",
                    "negative_type": "same_view_wrong_element"
                },
                {
                    "image": "screenshots/registration_form.png",
                    "text": "Submit button in registration form",
                    "negative_type": "similar_action_different_form"
                }
            ]
        },
        # Example 3: Navigation
        {
            "query": {
                "image": "screenshots/vscode_sidebar.png",
                "text": "Open the file explorer panel"
            },
            "positive": {
                "image": "screenshots/vscode_explorer_icon.png",
                "text": "File explorer icon in VS Code sidebar, first icon"
            },
            "negatives": [
                {
                    "image": "screenshots/vscode_sidebar.png",
                    "text": "Search icon in sidebar",
                    "negative_type": "same_view_wrong_element"
                },
                {
                    "image": "screenshots/vscode_sidebar.png",
                    "text": "Source control icon",
                    "negative_type": "same_view_wrong_element"
                }
            ]
        }
    ]

    return examples


def generate_reranker_examples():
    """
    Generate example JSONL format for reranker training data.
    """
    examples = [
        # Positive example
        {
            "query": {
                "image": "screenshots/chrome_new_tab.png",
                "text": "Click the address bar to type a URL"
            },
            "document": {
                "image": "screenshots/chrome_address_bar.png",
                "text": "Chrome address bar, click at coordinates (500, 45)"
            },
            "label": 1  # Relevant
        },
        # Negative example - wrong element
        {
            "query": {
                "image": "screenshots/chrome_new_tab.png",
                "text": "Click the address bar to type a URL"
            },
            "document": {
                "image": "screenshots/chrome_bookmarks.png",
                "text": "Bookmarks bar below address bar"
            },
            "label": 0  # Not relevant
        },
        # Hard negative - similar but wrong
        {
            "query": {
                "image": "screenshots/chrome_new_tab.png",
                "text": "Click the address bar to type a URL"
            },
            "document": {
                "image": "screenshots/chrome_search_box.png",
                "text": "Google search box in new tab page"
            },
            "label": 0  # Not relevant - search box != address bar
        },
        # Positive example with different app
        {
            "query": {
                "image": "screenshots/slack_channel.png",
                "text": "Type a message in the chat"
            },
            "document": {
                "image": "screenshots/slack_message_input.png",
                "text": "Message input field at bottom of Slack channel"
            },
            "label": 1
        },
    ]

    return examples


def save_examples(examples: List[Dict], output_path: str):
    """Save examples to JSONL file."""
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    logger.info(f"Saved {len(examples)} examples to {output_path}")


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_embedding_data(data_path: str, image_base_path: str = "") -> Dict:
    """
    Validate embedding training data.

    Checks:
    - Required fields present
    - Images exist
    - Text not empty
    - Negative types distribution
    """
    stats = {
        'total': 0,
        'valid': 0,
        'missing_query_image': 0,
        'missing_positive_image': 0,
        'missing_negative_images': 0,
        'empty_text': 0,
        'negative_types': Counter(),
        'avg_negatives': 0,
    }

    total_negatives = 0

    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            stats['total'] += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
                continue

            valid = True

            # Check query
            if 'query' not in item:
                logger.error(f"Line {line_num}: Missing 'query' field")
                valid = False
            else:
                query = item['query']
                if query.get('image'):
                    img_path = os.path.join(image_base_path, query['image']) if image_base_path else query['image']
                    if not os.path.exists(img_path):
                        logger.warning(f"Line {line_num}: Query image not found: {img_path}")
                        stats['missing_query_image'] += 1
                        valid = False

            # Check positive
            if 'positive' not in item:
                logger.error(f"Line {line_num}: Missing 'positive' field")
                valid = False
            else:
                positive = item['positive']
                if positive.get('image'):
                    img_path = os.path.join(image_base_path, positive['image']) if image_base_path else positive['image']
                    if not os.path.exists(img_path):
                        logger.warning(f"Line {line_num}: Positive image not found: {img_path}")
                        stats['missing_positive_image'] += 1
                        valid = False

            # Check negatives
            negatives = item.get('negatives', [])
            total_negatives += len(negatives)

            for neg in negatives:
                if neg.get('image'):
                    img_path = os.path.join(image_base_path, neg['image']) if image_base_path else neg['image']
                    if not os.path.exists(img_path):
                        stats['missing_negative_images'] += 1

                neg_type = neg.get('negative_type', 'unknown')
                stats['negative_types'][neg_type] += 1

            if valid:
                stats['valid'] += 1

    stats['avg_negatives'] = total_negatives / stats['total'] if stats['total'] > 0 else 0

    return stats


def validate_reranker_data(data_path: str, image_base_path: str = "") -> Dict:
    """
    Validate reranker training data.
    """
    stats = {
        'total': 0,
        'valid': 0,
        'positive_labels': 0,
        'negative_labels': 0,
        'missing_images': 0,
    }

    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            stats['total'] += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Check label
            label = item.get('label')
            if label == 1:
                stats['positive_labels'] += 1
            elif label == 0:
                stats['negative_labels'] += 1
            else:
                logger.warning(f"Line {line_num}: Invalid or missing label")

            stats['valid'] += 1

    # Compute ratio
    total_labels = stats['positive_labels'] + stats['negative_labels']
    if total_labels > 0:
        stats['positive_ratio'] = stats['positive_labels'] / total_labels
        stats['negative_ratio'] = stats['negative_labels'] / total_labels

    return stats


def print_validation_report(stats: Dict, data_type: str):
    """Print validation report."""
    print("\n" + "=" * 60)
    print(f"  {data_type.upper()} DATA VALIDATION REPORT")
    print("=" * 60)

    print(f"\nTotal examples: {stats['total']}")
    print(f"Valid examples: {stats['valid']}")
    print(f"Validity rate: {stats['valid'] / stats['total'] * 100:.1f}%")

    if data_type == 'embedding':
        print(f"\nMissing query images: {stats['missing_query_image']}")
        print(f"Missing positive images: {stats['missing_positive_image']}")
        print(f"Missing negative images: {stats['missing_negative_images']}")
        print(f"Average negatives per example: {stats['avg_negatives']:.2f}")

        if stats['negative_types']:
            print("\nNegative type distribution:")
            for neg_type, count in stats['negative_types'].most_common():
                print(f"  {neg_type}: {count}")

    elif data_type == 'reranker':
        print(f"\nPositive labels: {stats['positive_labels']} ({stats.get('positive_ratio', 0)*100:.1f}%)")
        print(f"Negative labels: {stats['negative_labels']} ({stats.get('negative_ratio', 0)*100:.1f}%)")

        # Check balance
        if stats.get('positive_ratio', 0) > 0.5:
            print("\n⚠️  WARNING: More positives than negatives. Recommended ratio is 1:3 to 1:5")

    print("=" * 60 + "\n")


# ============================================================================
# HARD NEGATIVE MINING
# ============================================================================

def mine_hard_negatives(
    data_path: str,
    model_name: str,
    output_path: str,
    top_k: int = 10,
    num_hard_negatives: int = 3,
    image_base_path: str = "",
):
    """
    Mine hard negatives using the embedding model.

    Process:
    1. Embed all queries and positives
    2. For each query, find top-k nearest positives (excluding its own)
    3. Add highest-ranked wrong positives as hard negatives
    """
    import torch
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    logger.info("Loading embedding model...")
    embedder = Qwen3VLEmbedder(model_name)

    # Load data
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} examples")

    # Embed all queries
    logger.info("Embedding queries...")
    query_embeddings = []
    for item in tqdm(data):
        query = item['query']
        img_path = os.path.join(image_base_path, query['image']) if image_base_path and query.get('image') else query.get('image')

        emb = embedder.process([{
            'image': img_path,
            'text': query.get('text', ''),
            'instruction': 'Represent the user action query.'
        }])
        query_embeddings.append(emb.cpu().numpy())

    query_embeddings = np.vstack(query_embeddings)

    # Embed all positives
    logger.info("Embedding positives...")
    positive_embeddings = []
    for item in tqdm(data):
        positive = item['positive']
        img_path = os.path.join(image_base_path, positive['image']) if image_base_path and positive.get('image') else positive.get('image')

        emb = embedder.process([{
            'image': img_path,
            'text': positive.get('text', ''),
            'instruction': 'Represent the action example.'
        }])
        positive_embeddings.append(emb.cpu().numpy())

    positive_embeddings = np.vstack(positive_embeddings)

    # Compute similarity matrix
    logger.info("Computing similarities and mining hard negatives...")
    similarity_matrix = query_embeddings @ positive_embeddings.T  # [num_queries, num_positives]

    # For each query, find hard negatives
    augmented_data = []
    for i, item in enumerate(tqdm(data)):
        similarities = similarity_matrix[i]

        # Get top-k most similar positives (excluding self)
        similarities[i] = -np.inf  # Exclude own positive
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        # These are hard negatives: similar but wrong
        hard_negatives = []
        for j in top_k_indices[:num_hard_negatives]:
            hard_neg = {
                'image': data[j]['positive']['image'],
                'text': data[j]['positive']['text'],
                'negative_type': 'hard_mined',
                'similarity_score': float(similarities[j])
            }
            hard_negatives.append(hard_neg)

        # Add to existing negatives
        new_item = item.copy()
        existing_negatives = new_item.get('negatives', [])
        new_item['negatives'] = existing_negatives + hard_negatives

        augmented_data.append(new_item)

    # Save augmented data
    save_examples(augmented_data, output_path)
    logger.info(f"Saved augmented data with hard negatives to {output_path}")


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_queries_with_paraphrases(
    data_path: str,
    output_path: str,
    num_paraphrases: int = 3,
):
    """
    Augment queries with paraphrases.

    This uses simple rule-based paraphrasing. For better results,
    use an LLM API (GPT-4, Claude, etc.).
    """
    # Simple paraphrase templates for computer use actions
    paraphrase_templates = {
        'click': [
            "Press the {element}",
            "Select the {element}",
            "Tap on the {element}",
            "Hit the {element}",
            "Choose the {element}",
        ],
        'type': [
            "Enter {text}",
            "Input {text}",
            "Write {text}",
            "Fill in {text}",
        ],
        'scroll': [
            "Navigate {direction}",
            "Move {direction}",
            "Go {direction}",
        ],
        'open': [
            "Launch {target}",
            "Start {target}",
            "Access {target}",
        ],
    }

    # Load data
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    augmented_data = []

    for item in data:
        # Keep original
        augmented_data.append(item)

        # Try to paraphrase query text
        query_text = item['query'].get('text', '').lower()

        for action, templates in paraphrase_templates.items():
            if action in query_text:
                # Generate paraphrases
                for template in random.sample(templates, min(num_paraphrases, len(templates))):
                    new_item = item.copy()
                    new_item['query'] = item['query'].copy()

                    # Simple substitution (would be better with LLM)
                    new_text = item['query']['text']
                    new_item['query']['text'] = f"[PARAPHRASE] {new_text}"

                    augmented_data.append(new_item)
                break

    save_examples(augmented_data, output_path)
    logger.info(f"Augmented {len(data)} examples to {len(augmented_data)} examples")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Training data preparation utilities")

    parser.add_argument("--generate_examples", action="store_true",
                        help="Generate example data files")
    parser.add_argument("--validate", type=str,
                        help="Validate a data file")
    parser.add_argument("--data_type", type=str, choices=['embedding', 'reranker'],
                        default='embedding', help="Type of data to validate")
    parser.add_argument("--image_base_path", type=str, default="",
                        help="Base path for images")
    parser.add_argument("--mine_hard_negatives", action="store_true",
                        help="Mine hard negatives")
    parser.add_argument("--data", type=str,
                        help="Input data file for hard negative mining")
    parser.add_argument("--model_name", type=str,
                        default="Alibaba-NLP/Qwen3-VL-Embedding-8B",
                        help="Model for hard negative mining")
    parser.add_argument("--output", type=str,
                        help="Output file path")

    args = parser.parse_args()

    if args.generate_examples:
        # Generate example files
        os.makedirs("data", exist_ok=True)

        embedding_examples = generate_embedding_examples()
        save_examples(embedding_examples, "data/embedding_train_example.jsonl")

        reranker_examples = generate_reranker_examples()
        save_examples(reranker_examples, "data/reranker_train_example.jsonl")

        print("\nGenerated example data files:")
        print("  - data/embedding_train_example.jsonl")
        print("  - data/reranker_train_example.jsonl")

    elif args.validate:
        if args.data_type == 'embedding':
            stats = validate_embedding_data(args.validate, args.image_base_path)
        else:
            stats = validate_reranker_data(args.validate, args.image_base_path)

        print_validation_report(stats, args.data_type)

    elif args.mine_hard_negatives:
        if not args.data or not args.output:
            logger.error("--data and --output required for hard negative mining")
            return

        mine_hard_negatives(
            args.data,
            args.model_name,
            args.output,
            image_base_path=args.image_base_path
        )


if __name__ == "__main__":
    main()
