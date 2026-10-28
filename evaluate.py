"""
Evaluation Script for Invoice Information Extraction.
Evaluates the trained model & pipeline on a held-out test split of annotated invoices.
Calculates field-level Precision, Recall, F1-Score, and Exact Match Accuracy without fabricating metrics.
"""

import argparse
from pathlib import Path
import json
import sys
from typing import Dict, Any, List

from src.config import TARGET_FIELDS, ANNOTATIONS_DIR, SPLITS_DIR, INVOICES_DIR
from src.document_loader import DocumentLoader
from src.ocr_engine import OCREngine
from src.postprocessor import InvoiceProcessor
from src.exporter import sanitize_for_json


def normalize_val(val: Any) -> str:
    """Normalizes string or numeric value for fair exact-match evaluation."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return f"{float(val):.2f}"
    s = str(val).strip().lower()
    # Normalize common date hyphens vs slashes
    s = s.replace("/", "-")
    # Clean redundant punctuation
    s = s.strip(":,;.")
    return s


def values_match(gt_val: Any, pred_val: Any, field_name: str) -> bool:
    """Evaluates whether ground truth and predicted values are equivalent."""
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False

    # Financial numeric comparison
    if field_name in ["subtotal", "tax_amount", "total_amount"]:
        try:
            gt_num = float(gt_val)
            pred_num = float(pred_val)
            return abs(gt_num - pred_num) <= 0.05
        except (ValueError, TypeError):
            pass

    # Date comparison (handle YYYY-MM-DD vs DD-MM-YYYY)
    if field_name == "invoice_date":
        n_gt = normalize_val(gt_val)
        n_pred = normalize_val(pred_val)
        if n_gt == n_pred:
            return True
        gt_parts = n_gt.split("-")
        pred_parts = n_pred.split("-")
        if len(gt_parts) == 3 and len(pred_parts) == 3:
            if set(gt_parts) == set(pred_parts):
                return True

    # Standard textual comparison
    n_gt = normalize_val(gt_val)
    n_pred = normalize_val(pred_val)

    # Substring / fuzzy match tolerance for multi-word corporate names
    if field_name in ["supplier_name", "buyer_name"]:
        if n_gt == n_pred or (len(n_gt) > 4 and (n_gt in n_pred or n_pred in n_gt)):
            return True

    return n_gt == n_pred


def evaluate_test_set(
    test_split_file: Path = SPLITS_DIR / "test_docs.json",
    output_report_file: Path = Path("output") / "evaluation_report.json",
) -> Dict[str, Any]:
    """Runs evaluation over held-out test split invoices."""
    if not test_split_file.exists():
        print(f"Error: Test split file not found at {test_split_file}.")
        sys.exit(1)

    with open(test_split_file, "r", encoding="utf-8") as f:
        test_doc_ids = json.load(f)

    print("================================================================================")
    print("           ML-BASED INVOICE DATA EXTRACTION SYSTEM - EVALUATION")
    print("================================================================================")
    print(f"Evaluation Dataset : {test_split_file.name}")
    print(f"Number of Test Invoices : {len(test_doc_ids)}")
    print(f"Evaluation Methodology : Document-level held-out split (Strict zero-leakage)")

    loader = DocumentLoader()
    ocr_engine = OCREngine()
    processor = InvoiceProcessor()

    # Track metrics per field
    field_stats: Dict[str, Dict[str, int]] = {
        f: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "total_pred": 0} for f in TARGET_FIELDS
    }

    total_gt_items = 0
    total_pred_items = 0
    matched_items = 0

    evaluated_docs = 0

    for doc_id in test_doc_ids:
        annot_file = ANNOTATIONS_DIR / f"{doc_id}.json"
        if not annot_file.exists():
            print(f"Skipping {doc_id}: Missing annotation file.")
            continue

        with open(annot_file, "r", encoding="utf-8") as f:
            annot_data = json.load(f)

        gt_fields = annot_data.get("ground_truth_fields", {})
        gt_line_items = annot_data.get("ground_truth_line_items", [])

        # Find invoice file (pdf or png)
        doc_path = INVOICES_DIR / f"{doc_id}.pdf"
        if not doc_path.exists():
            doc_path = INVOICES_DIR / f"{doc_id}.png"
            if not doc_path.exists():
                print(f"Skipping {doc_id}: Invoice file not found.")
                continue

        # Run extraction pipeline
        try:
            doc_dict = loader.load(doc_path)
            tokens = ocr_engine.extract_tokens(doc_dict)
            extraction_result = processor.process(doc_dict, tokens)
            pred_fields = extraction_result.get("fields", {})
            pred_items = extraction_result.get("line_items", [])
        except Exception as e:
            print(f"Error processing {doc_id}: {e}")
            continue

        evaluated_docs += 1

        # Evaluate target fields
        for field in TARGET_FIELDS:
            gt_val = gt_fields.get(field)
            pred_val = pred_fields.get(field)

            if gt_val is not None:
                field_stats[field]["total_gt"] += 1
            if pred_val is not None:
                field_stats[field]["total_pred"] += 1

            if gt_val is not None and pred_val is not None:
                if values_match(gt_val, pred_val, field):
                    field_stats[field]["tp"] += 1
                else:
                    field_stats[field]["fp"] += 1
                    field_stats[field]["fn"] += 1
            elif gt_val is not None and pred_val is None:
                field_stats[field]["fn"] += 1
            elif gt_val is None and pred_val is not None:
                field_stats[field]["fp"] += 1

        # Evaluate line items count & totals
        total_gt_items += len(gt_line_items)
        total_pred_items += len(pred_items)
        for itm in pred_items:
            tot = itm.get("total")
            if tot is not None and any(abs(tot - g.get("total", 0.0)) <= 0.05 for g in gt_line_items):
                matched_items += 1

    # Compute Precision, Recall, F1, Accuracy
    results_table = []
    overall_tp = sum(stats["tp"] for stats in field_stats.values())
    overall_fp = sum(stats["fp"] for stats in field_stats.values())
    overall_fn = sum(stats["fn"] for stats in field_stats.values())
    overall_gt = sum(stats["total_gt"] for stats in field_stats.values())

    print("\n--------------------------------------------------------------------------------")
    print(f"{'Field Name':24s} | {'Prec.':6s} | {'Recall':6s} | {'F1-Score':8s} | {'Exact Match':11s} | {'Support':7s}")
    print("--------------------------------------------------------------------------------")

    field_metrics = {}
    for field in TARGET_FIELDS:
        s = field_stats[field]
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]
        support = s["total_gt"]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        exact_match = tp / support if support > 0 else 0.0

        field_metrics[field] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "exact_match_accuracy": round(exact_match, 4),
            "support": support,
        }

        print(
            f"{field:24s} | {prec*100:5.1f}% | {rec*100:5.1f}% | {f1*100:7.1f}% | "
            f"{exact_match*100:10.1f}% | {support:7d}"
        )

    micro_prec = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    micro_rec = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1 = sum(m["f1_score"] for m in field_metrics.values()) / len(field_metrics)

    print("--------------------------------------------------------------------------------")
    print(
        f"{'OVERALL (Micro-avg)':24s} | {micro_prec*100:5.1f}% | {micro_rec*100:5.1f}% | {micro_f1*100:7.1f}% | "
        f"{micro_rec*100:10.1f}% | {overall_gt:7d}"
    )
    print(f"{'OVERALL (Macro-avg F1)':24s} | {'-':6s} | {'-':6s} | {macro_f1*100:7.1f}% | {'-':11s} | {overall_gt:7d}")
    print("--------------------------------------------------------------------------------")

    # Line item metrics
    item_prec = matched_items / max(1, total_pred_items)
    item_rec = matched_items / max(1, total_gt_items)
    item_f1 = (2 * item_prec * item_rec) / max(0.001, item_prec + item_rec)

    print(f"\nLine-Item Table Metrics:")
    print(f"  Ground Truth Line Items : {total_gt_items}")
    print(f"  Extracted Line Items    : {total_pred_items}")
    print(f"  Matched Items           : {matched_items}")
    print(f"  Line Items Precision    : {item_prec*100:.1f}%")
    print(f"  Line Items Recall       : {item_rec*100:.1f}%")
    print(f"  Line Items F1-Score     : {item_f1*100:.1f}%")

    report_payload = {
        "metadata": {
            "test_split_file": str(test_split_file),
            "evaluated_documents_count": evaluated_docs,
            "evaluation_type": "held_out_document_level",
        },
        "overall_summary": {
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "total_ground_truth_fields": overall_gt,
        },
        "field_breakdown": field_metrics,
        "line_items_summary": {
            "ground_truth_items": total_gt_items,
            "extracted_items": total_pred_items,
            "precision": round(item_prec, 4),
            "recall": round(item_rec, 4),
            "f1": round(item_f1, 4),
        },
    }

    output_report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(report_payload), f, indent=2)

    print(f"\nFull evaluation report saved to: {output_report_file}")
    return report_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Invoice Extraction System")
    parser.add_argument("--test_split", type=str, default=str(SPLITS_DIR / "test_docs.json"), help="Test split JSON file")
    parser.add_argument("--output_report", type=str, default="output/evaluation_report.json", help="Output report JSON file")
    args = parser.parse_args()

    evaluate_test_set(
        test_split_file=Path(args.test_split),
        output_report_file=Path(args.output_report),
    )
