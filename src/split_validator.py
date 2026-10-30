"""
Split Verification and Leakage Audit Module.
Validates train/test split files for disjointness, existence, integrity, and zero leakage.
"""

from typing import Dict, Any, List, Set, Tuple
from pathlib import Path
import json

from src.config import SPLITS_DIR, INVOICES_DIR, ANNOTATIONS_DIR


def verify_splits(
    train_split_path: Path = SPLITS_DIR / "train_docs.json",
    test_split_path: Path = SPLITS_DIR / "test_docs.json",
    invoices_dir: Path = INVOICES_DIR,
    annotations_dir: Path = ANNOTATIONS_DIR,
) -> Dict[str, Any]:
    """
    Validates that:
    1. Both split files exist and contain valid lists of document IDs.
    2. Train and test sets are strictly disjoint (no overlapping documents).
    3. No duplicate document IDs exist within either split.
    4. Every document ID references an existing invoice file (.pdf or .png).
    5. Every document ID references an existing ground-truth annotation file (.json).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not train_split_path.exists():
        errors.append(f"Train split file not found: {train_split_path}")
        train_list = []
    else:
        with open(train_split_path, "r", encoding="utf-8") as f:
            train_list = json.load(f)

    if not test_split_path.exists():
        errors.append(f"Test split file not found: {test_split_path}")
        test_list = []
    else:
        with open(test_split_path, "r", encoding="utf-8") as f:
            test_list = json.load(f)

    # 1. Duplicates within splits
    train_duplicates = [d for d in set(train_list) if train_list.count(d) > 1]
    test_duplicates = [d for d in set(test_list) if test_list.count(d) > 1]

    if train_duplicates:
        errors.append(f"Duplicate document IDs found in train split: {train_duplicates}")
    if test_duplicates:
        errors.append(f"Duplicate document IDs found in test split: {test_duplicates}")

    train_set = set(train_list)
    test_set = set(test_list)

    # 2. Overlap / Data Leakage Check
    overlap = train_set.intersection(test_set)
    if overlap:
        errors.append(f"CRITICAL DATA LEAKAGE: Overlapping document IDs in train and test sets: {sorted(list(overlap))}")

    # 3. Document and Annotation Existence Checks
    all_docs = train_set.union(test_set)
    missing_invoices = []
    missing_annotations = []

    for doc_id in sorted(list(all_docs)):
        pdf_file = invoices_dir / f"{doc_id}.pdf"
        png_file = invoices_dir / f"{doc_id}.png"
        jpg_file = invoices_dir / f"{doc_id}.jpg"
        if not (pdf_file.exists() or png_file.exists() or jpg_file.exists()):
            missing_invoices.append(doc_id)

        annot_file = annotations_dir / f"{doc_id}.json"
        if not annot_file.exists():
            missing_annotations.append(doc_id)

    if missing_invoices:
        errors.append(f"Missing invoice document files for: {missing_invoices}")
    if missing_annotations:
        errors.append(f"Missing ground-truth annotation files for: {missing_annotations}")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "train_count": len(train_list),
        "test_count": len(test_list),
        "total_unique_documents": len(all_docs),
        "train_doc_ids": sorted(list(train_set)),
        "test_doc_ids": sorted(list(test_set)),
        "overlap_count": len(overlap),
        "overlap_doc_ids": sorted(list(overlap)),
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    result = verify_splits()
    print("Split Verification Result:", json.dumps(result, indent=2))
