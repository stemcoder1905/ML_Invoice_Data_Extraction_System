"""
CRF Model Training Script for Invoice Extraction.
Loads annotated training documents (grouped strictly by document to prevent leakage),
computes lexical/spatial/contextual features, trains the CRF model, and saves the artifact.
"""

import argparse
from pathlib import Path
import json
from collections import Counter
import sys

from src.config import ANNOTATIONS_DIR, SPLITS_DIR, DEFAULT_MODEL_PATH
from src.feature_engineering import FeatureExtractor
from src.ml_extractor import InvoiceCRFExtractor


def train_model(
    split_file: Path = SPLITS_DIR / "train_docs.json",
    model_save_path: Path = DEFAULT_MODEL_PATH,
    c1: float = 0.1,
    c2: float = 0.1,
    max_iter: int = 100,
) -> None:
    """Trains CRF sequence tagger strictly on designated training split documents."""
    if not split_file.exists():
        print(f"Error: Train split file not found at {split_file}. Run 'python generate_dataset.py' first.")
        sys.exit(1)

    with open(split_file, "r", encoding="utf-8") as f:
        train_doc_ids = json.load(f)

    print(f"=== Starting CRF Model Training ===")
    print(f"Split file: {split_file}")
    print(f"Number of training documents: {len(train_doc_ids)}")
    print(f"Model destination: {model_save_path}")

    feature_extractor = FeatureExtractor()
    X_train = []
    y_train = []
    label_counter = Counter()

    for doc_id in train_doc_ids:
        annot_file = ANNOTATIONS_DIR / f"{doc_id}.json"
        if not annot_file.exists():
            print(f"Warning: Annotation file {annot_file} not found, skipping.")
            continue

        with open(annot_file, "r", encoding="utf-8") as f:
            annot_data = json.load(f)

        tokens = annot_data.get("tokens", [])
        if not tokens:
            continue

        seq_features = feature_extractor.extract_sequence_features(tokens)
        seq_labels = [t.get("bio_tag", "O") for t in tokens]

        X_train.append(seq_features)
        y_train.append(seq_labels)
        label_counter.update(seq_labels)

    if not X_train:
        print("Error: No training sequences extracted. Training aborted.")
        sys.exit(1)

    total_tokens = sum(len(seq) for seq in X_train)
    print(f"\nTraining Dataset Statistics:")
    print(f"Total training tokens: {total_tokens}")
    print(f"Label distribution:")
    for lbl, count in sorted(label_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:15s}: {count:5d} tokens")

    print(f"\nTraining CRF sequence tagger (algorithm='lbfgs', c1={c1}, c2={c2}, max_iter={max_iter})...")
    extractor = InvoiceCRFExtractor(model_path=model_save_path)
    train_info = extractor.train(X_train, y_train, c1=c1, c2=c2, max_iterations=max_iter)

    print(f"\nModel training completed successfully!")
    print(f"Saved CRF model to: {train_info['model_path']}")
    print(f"Total classes learned: {len(train_info['classes'])}")
    print(f"Learned classes: {', '.join(train_info['classes'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRF Model on Annotated Invoices")
    parser.add_argument("--split", type=str, default=str(SPLITS_DIR / "train_docs.json"), help="Path to train split JSON")
    parser.add_argument("--output", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to save trained CRF model")
    parser.add_argument("--c1", type=float, default=0.1, help="L1 regularization parameter")
    parser.add_argument("--c2", type=float, default=0.1, help="L2 regularization parameter")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum L-BFGS iterations")
    args = parser.parse_args()

    train_model(
        split_file=Path(args.split),
        model_save_path=Path(args.output),
        c1=args.c1,
        c2=args.c2,
        max_iter=args.max_iter,
    )
