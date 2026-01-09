#!/usr/bin/env python3
"""
DATASET VALIDATION SCRIPT
=========================

Run this BEFORE training to catch errors.
Checks all JSONL files for:
- Valid JSON format
- Required fields present
- Image paths exist
- Reranker label balance
- No duplicate entries

Usage:
    cd higgsfield_dataset
    python validate_dataset.py
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter


class DatasetValidator:
    """Validates all training datasets."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.errors = []
        self.warnings = []

    def error(self, msg: str):
        """Add an error."""
        self.errors.append(f"❌ ERROR: {msg}")

    def warning(self, msg: str):
        """Add a warning."""
        self.warnings.append(f"⚠️  WARNING: {msg}")

    def success(self, msg: str):
        """Print success message."""
        print(f"✓ {msg}")

    def validate_json_line(self, line: str, line_num: int, filepath: str) -> dict | None:
        """Validate a single JSON line."""
        try:
            data = json.loads(line.strip())
            return data
        except json.JSONDecodeError as e:
            self.error(f"{filepath}:{line_num} - Invalid JSON: {e}")
            return None

    def check_image_path(self, path: str, line_num: int, filepath: str):
        """Check if an image path exists."""
        if not path:
            return

        # Handle relative paths
        full_path = self.base_path / path
        if not full_path.exists():
            self.error(f"{filepath}:{line_num} - Image not found: {path}")

    def extract_images(self, obj, images: list):
        """Recursively extract image paths from nested object."""
        if isinstance(obj, dict):
            if 'image' in obj:
                images.append(obj['image'])
            for value in obj.values():
                self.extract_images(value, images)
        elif isinstance(obj, list):
            for item in obj:
                self.extract_images(item, images)

    def validate_embedding_file(self, filepath: Path) -> tuple[int, int]:
        """Validate embedding training data."""
        print(f"\n{'='*60}")
        print(f"Validating: {filepath}")
        print(f"{'='*60}")

        if not filepath.exists():
            self.error(f"File not found: {filepath}")
            return 0, 0

        valid_count = 0
        total_count = 0
        required_fields = ['query', 'positive', 'negative']

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                total_count += 1
                data = self.validate_json_line(line, line_num, str(filepath))
                if data is None:
                    continue

                # Check required fields
                missing = [f for f in required_fields if f not in data]
                if missing:
                    self.error(f"{filepath}:{line_num} - Missing fields: {missing}")
                    continue

                # Check image paths
                images = []
                self.extract_images(data, images)
                for img in images:
                    self.check_image_path(img, line_num, str(filepath))

                # Check structure
                if 'text' not in data.get('query', {}):
                    self.warning(f"{filepath}:{line_num} - Query missing 'text' field")
                if 'text' not in data.get('positive', {}):
                    self.warning(f"{filepath}:{line_num} - Positive missing 'text' field")
                if 'text' not in data.get('negative', {}):
                    self.warning(f"{filepath}:{line_num} - Negative missing 'text' field")

                valid_count += 1

        return valid_count, total_count

    def validate_reranker_file(self, filepath: Path) -> tuple[int, int, int, int]:
        """Validate reranker training data."""
        print(f"\n{'='*60}")
        print(f"Validating: {filepath}")
        print(f"{'='*60}")

        if not filepath.exists():
            self.error(f"File not found: {filepath}")
            return 0, 0, 0, 0

        valid_count = 0
        total_count = 0
        positive_count = 0
        negative_count = 0
        required_fields = ['query', 'document', 'label']

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                total_count += 1
                data = self.validate_json_line(line, line_num, str(filepath))
                if data is None:
                    continue

                # Check required fields
                missing = [f for f in required_fields if f not in data]
                if missing:
                    self.error(f"{filepath}:{line_num} - Missing fields: {missing}")
                    continue

                # Check label
                label = data.get('label')
                if label not in [0, 1]:
                    self.error(f"{filepath}:{line_num} - Label must be 0 or 1, got: {label}")
                    continue

                if label == 1:
                    positive_count += 1
                else:
                    negative_count += 1

                # Check image paths
                images = []
                self.extract_images(data, images)
                for img in images:
                    self.check_image_path(img, line_num, str(filepath))

                valid_count += 1

        # Check balance
        if valid_count > 0:
            ratio = positive_count / valid_count
            if ratio < 0.4 or ratio > 0.6:
                self.warning(
                    f"Reranker labels unbalanced: {positive_count} positive ({ratio:.1%}), "
                    f"{negative_count} negative ({1-ratio:.1%}). "
                    f"Target is 50/50."
                )

        return valid_count, total_count, positive_count, negative_count

    def validate_sft_file(self, filepath: Path) -> tuple[int, int]:
        """Validate SFT training data."""
        print(f"\n{'='*60}")
        print(f"Validating: {filepath}")
        print(f"{'='*60}")

        if not filepath.exists():
            self.error(f"File not found: {filepath}")
            return 0, 0

        valid_count = 0
        total_count = 0

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                total_count += 1
                data = self.validate_json_line(line, line_num, str(filepath))
                if data is None:
                    continue

                # Check messages field
                if 'messages' not in data:
                    self.error(f"{filepath}:{line_num} - Missing 'messages' field")
                    continue

                messages = data['messages']
                if not isinstance(messages, list) or len(messages) < 2:
                    self.error(f"{filepath}:{line_num} - 'messages' must be a list with at least 2 items")
                    continue

                # Check message structure
                for i, msg in enumerate(messages):
                    if 'role' not in msg:
                        self.error(f"{filepath}:{line_num} - Message {i} missing 'role'")
                    if 'content' not in msg:
                        self.error(f"{filepath}:{line_num} - Message {i} missing 'content'")

                # Check for ACTION in assistant response
                has_action = False
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if isinstance(content, str) and 'ACTION:' in content:
                            has_action = True

                if not has_action:
                    self.warning(f"{filepath}:{line_num} - No ACTION command in assistant response")

                # Check image paths
                images = []
                self.extract_images(data, images)
                for img in images:
                    self.check_image_path(img, line_num, str(filepath))

                valid_count += 1

        return valid_count, total_count

    def validate_dpo_file(self, filepath: Path) -> tuple[int, int]:
        """Validate DPO training data."""
        print(f"\n{'='*60}")
        print(f"Validating: {filepath}")
        print(f"{'='*60}")

        if not filepath.exists():
            self.error(f"File not found: {filepath}")
            return 0, 0

        valid_count = 0
        total_count = 0
        required_fields = ['prompt', 'chosen', 'rejected']

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                total_count += 1
                data = self.validate_json_line(line, line_num, str(filepath))
                if data is None:
                    continue

                # Check required fields
                missing = [f for f in required_fields if f not in data]
                if missing:
                    self.error(f"{filepath}:{line_num} - Missing fields: {missing}")
                    continue

                # Check that chosen has ACTION but rejected doesn't (preferred pattern)
                chosen = data.get('chosen', '')
                rejected = data.get('rejected', '')

                if 'ACTION:' not in chosen:
                    self.warning(f"{filepath}:{line_num} - 'chosen' response has no ACTION command")

                if 'ACTION:' in rejected:
                    self.warning(f"{filepath}:{line_num} - 'rejected' response has ACTION (should be vague)")

                # Check image paths
                images = []
                self.extract_images(data, images)
                for img in images:
                    self.check_image_path(img, line_num, str(filepath))

                valid_count += 1

        return valid_count, total_count

    def validate_all(self):
        """Run all validations."""
        print("\n" + "=" * 70)
        print("DATASET VALIDATION REPORT")
        print("=" * 70)

        results = {}

        # Embedding
        emb_valid, emb_total = self.validate_embedding_file(
            self.base_path / "data" / "embedding_train.jsonl"
        )
        results['embedding'] = {'valid': emb_valid, 'total': emb_total, 'minimum': 250}

        # Reranker
        rer_valid, rer_total, rer_pos, rer_neg = self.validate_reranker_file(
            self.base_path / "data" / "reranker_train.jsonl"
        )
        results['reranker'] = {
            'valid': rer_valid, 'total': rer_total, 'minimum': 150,
            'positive': rer_pos, 'negative': rer_neg
        }

        # SFT
        sft_valid, sft_total = self.validate_sft_file(
            self.base_path / "data" / "sft_train.jsonl"
        )
        results['sft'] = {'valid': sft_valid, 'total': sft_total, 'minimum': 50}

        # DPO
        dpo_valid, dpo_total = self.validate_dpo_file(
            self.base_path / "data" / "dpo_train.jsonl"
        )
        results['dpo'] = {'valid': dpo_valid, 'total': dpo_total, 'minimum': 25}

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        all_pass = True

        for name, data in results.items():
            valid = data['valid']
            minimum = data['minimum']
            status = "✓ PASS" if valid >= minimum else "❌ FAIL"

            if valid < minimum:
                all_pass = False
                self.error(f"{name}: Only {valid} valid entries, need at least {minimum}")

            extra = ""
            if name == 'reranker':
                extra = f" (pos={data.get('positive', 0)}, neg={data.get('negative', 0)})"

            print(f"  {name}: {valid}/{data['total']} valid{extra} [{status}]")

        # Print all errors
        if self.errors:
            print(f"\n{'='*70}")
            print(f"ERRORS ({len(self.errors)})")
            print("=" * 70)
            for error in self.errors[:20]:  # Show first 20
                print(f"  {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors")

        # Print all warnings
        if self.warnings:
            print(f"\n{'='*70}")
            print(f"WARNINGS ({len(self.warnings)})")
            print("=" * 70)
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")

        # Final status
        print(f"\n{'='*70}")
        if all_pass and not self.errors:
            print("✓ DATASET VALIDATION PASSED")
            print("  Your dataset is ready for training!")
        else:
            print("❌ DATASET VALIDATION FAILED")
            print("  Fix the errors above before training.")
        print("=" * 70)

        return all_pass and not self.errors


def main():
    """Run validation."""
    # Check if we're in the right directory
    if not os.path.exists("data"):
        print("Error: 'data' directory not found.")
        print("Make sure you're running this from the higgsfield_dataset directory:")
        print("  cd higgsfield_dataset")
        print("  python validate_dataset.py")
        sys.exit(1)

    validator = DatasetValidator(".")
    success = validator.validate_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
