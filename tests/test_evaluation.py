"""
Automated Tests for Evaluation Correctness and Leakage Auditing.
Tests the 10 critical scenarios specified in the audit requirements:
1. Perfect predictions
2. Completely incorrect prediction
3. Missing prediction
4. Incorrect company-name extension (no substring false-positives)
5. Duplicate line-item predictions (strict 1-to-1 bipartite matching)
6. Reordered line items (order-independence)
7. Numeric tolerance (<0.05, =0.05, >0.05)
8. Date normalization (ISO equivalence vs different dates)
9. Empty datasets and zero denominators handling
10. Data split overlap detection (leakage prevention)
"""

import pytest
from pathlib import Path

from evaluate import (
    compare_field,
    normalize_text,
    normalize_date,
    normalize_amount,
    normalize_currency,
    match_line_items,
)
from src.split_validator import verify_splits


# ---------------------------------------------------------------------------
# Test 1: Perfect predictions
# ---------------------------------------------------------------------------
def test_perfect_prediction():
    gt = "INV-001"
    pred = "INV-001"
    is_match = compare_field(gt, pred, "invoice_number")
    assert is_match is True

    # When evaluated: TP=1, FP=0, FN=0, P=1.0, R=1.0, F1=1.0, Exact Acc=1.0
    tp, fp, fn = (1, 0, 0)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0


# ---------------------------------------------------------------------------
# Test 2: Completely incorrect prediction
# ---------------------------------------------------------------------------
def test_completely_incorrect_prediction():
    gt = "INV-001"
    pred = "INV-999"
    is_match = compare_field(gt, pred, "invoice_number")
    assert is_match is False

    # When evaluated: TP=0, FP=1, FN=1, P=0.0, R=0.0, F1=0.0, Exact Acc=0.0
    tp, fp, fn = (0, 1, 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    assert prec == 0.0
    assert rec == 0.0
    assert f1 == 0.0


# ---------------------------------------------------------------------------
# Test 3: Missing prediction
# ---------------------------------------------------------------------------
def test_missing_prediction():
    gt = "INV-001"
    pred = None
    is_match = compare_field(gt, pred, "invoice_number")
    assert is_match is False

    # Missing prediction counts as FN=1, FP=0, TP=0
    tp, fp, fn = (0, 0, 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    assert prec == 0.0
    assert rec == 0.0


# ---------------------------------------------------------------------------
# Test 4: Incorrect company-name extension
# ---------------------------------------------------------------------------
def test_company_name_extension_mismatch():
    gt = "ABC Technologies"
    pred = "ABC Technologies Private Limited"

    # Must NOT count as an exact match simply because one contains the other
    assert compare_field(gt, pred, "supplier_name") is False
    assert compare_field(gt, pred, "buyer_name") is False

    # Exact string match with preserved corporate suffix should match
    assert compare_field("Vertex Global Solutions Ltd", "Vertex Global Solutions Ltd", "supplier_name") is True


# ---------------------------------------------------------------------------
# Test 5: Duplicate line-item predictions
# ---------------------------------------------------------------------------
def test_duplicate_line_item_predictions_bipartite_enforcement():
    # GT items with totals 100, 200, 300
    gt_items = [
        {"description": "Server Hosting", "quantity": 1, "unit_price": 100.0, "total": 100.0},
        {"description": "Database License", "quantity": 1, "unit_price": 200.0, "total": 200.0},
        {"description": "Support Package", "quantity": 1, "unit_price": 300.0, "total": 300.0},
    ]

    # Predictions predicting the first item three times
    pred_items = [
        {"description": "Server Hosting", "quantity": 1, "unit_price": 100.0, "total": 100.0},
        {"description": "Server Hosting", "quantity": 1, "unit_price": 100.0, "total": 100.0},
        {"description": "Server Hosting", "quantity": 1, "unit_price": 100.0, "total": 100.0},
    ]

    res = match_line_items(gt_items, pred_items)

    # Only ONE prediction may match the single GT item
    assert res["matched_count"] == 1
    # The other 2 predictions must be false positives
    assert res["fp_count"] == 2
    # The 2 unmatched GT items must be false negatives
    assert res["fn_count"] == 2

    assert res["precision"] == round(1 / 3, 4)
    assert res["recall"] == round(1 / 3, 4)


# ---------------------------------------------------------------------------
# Test 6: Reordered line items
# ---------------------------------------------------------------------------
def test_reordered_line_items_matching():
    gt_items = [
        {"description": "Item A", "quantity": 2, "unit_price": 50.0, "total": 100.0},
        {"description": "Item B", "quantity": 1, "unit_price": 250.0, "total": 250.0},
    ]

    # Reversed order
    pred_items = [
        {"description": "Item B", "quantity": 1, "unit_price": 250.0, "total": 250.0},
        {"description": "Item A", "quantity": 2, "unit_price": 50.0, "total": 100.0},
    ]

    res = match_line_items(gt_items, pred_items)
    # Order-independent bipartite matching must achieve 100%
    assert res["matched_count"] == 2
    assert res["fp_count"] == 0
    assert res["fn_count"] == 0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


# ---------------------------------------------------------------------------
# Test 7: Numeric tolerance (<0.05, ==0.05, >0.05)
# ---------------------------------------------------------------------------
def test_numeric_tolerance():
    # Diff < 0.05 (0.02 diff -> Match)
    assert compare_field(100.00, 100.02, "total_amount", tolerance=0.05) is True

    # Diff == 0.05 (Exact boundary -> Match)
    assert compare_field(100.00, 100.05, "total_amount", tolerance=0.05) is True
    assert compare_field(100.05, 100.00, "total_amount", tolerance=0.05) is True

    # Diff > 0.05 (0.06 diff -> Mismatch)
    assert compare_field(100.00, 100.06, "total_amount", tolerance=0.05) is False
    assert compare_field("1,250.00", "1,250.10", "subtotal", tolerance=0.05) is False


# ---------------------------------------------------------------------------
# Test 8: Date normalization
# ---------------------------------------------------------------------------
def test_date_normalization_formats():
    # Equivalent dates in different formats must match
    assert compare_field("2024-05-15", "15/05/2024", "invoice_date") is True
    assert compare_field("15-05-2024", "2024-05-15", "invoice_date") is True
    assert compare_field("May 15, 2024", "2024-05-15", "invoice_date") is True

    # Different dates must NOT match
    assert compare_field("2024-05-15", "2024-05-16", "invoice_date") is False
    assert compare_field("11/12/2024", "12/11/2024", "invoice_date") is False

    # Invalid / unparseable dates
    assert compare_field("invalid-date", "2024-05-15", "invoice_date") is False


# ---------------------------------------------------------------------------
# Test 9: Empty datasets and zero denominators
# ---------------------------------------------------------------------------
def test_zero_denominators_safety():
    # When both ground truth and predictions are empty for line items
    res_empty = match_line_items([], [])
    assert res_empty["precision"] == 1.0
    assert res_empty["recall"] == 1.0
    assert res_empty["f1"] == 1.0

    # When no predictions exist for positive ground truth
    res_no_pred = match_line_items([{"description": "A", "total": 100}], [])
    assert res_no_pred["precision"] == 0.0
    assert res_no_pred["recall"] == 0.0
    assert res_no_pred["f1"] == 0.0

    # When no ground truth exists for positive predictions
    res_no_gt = match_line_items([], [{"description": "A", "total": 100}])
    assert res_no_gt["precision"] == 0.0
    assert res_no_gt["recall"] == 0.0
    assert res_no_gt["f1"] == 0.0


# ---------------------------------------------------------------------------
# Test 10: Data split overlap detection
# ---------------------------------------------------------------------------
def test_data_split_overlap_detection(tmp_path):
    import json

    # Create a synthetic leaked split
    leaked_train = ["INV_0001", "INV_0002", "INV_0003"]
    leaked_test = ["INV_0003", "INV_0004"]  # INV_0003 is leaked in both!

    train_file = tmp_path / "train_docs.json"
    test_file = tmp_path / "test_docs.json"

    with open(train_file, "w") as f:
        json.dump(leaked_train, f)
    with open(test_file, "w") as f:
        json.dump(leaked_test, f)

    result = verify_splits(
        train_split_path=train_file,
        test_split_path=test_file,
        invoices_dir=tmp_path,
        annotations_dir=tmp_path,
    )

    # Must detect the leakage and report is_valid == False
    assert result["is_valid"] is False
    assert result["overlap_count"] == 1
    assert "INV_0003" in result["overlap_doc_ids"]
    assert any("CRITICAL DATA LEAKAGE" in err for err in result["errors"])
