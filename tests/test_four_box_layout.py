"""
Tests for 4-box grid layouts and regression tests for party extraction,
duplicate buyer names, invoice number blacklist, and table tax calculations.
"""

from pathlib import Path
import json
import pytest

from src.document_loader import DocumentLoader
from src.ocr_engine import OCREngine
from src.postprocessor import InvoiceProcessor


@pytest.fixture(scope="module")
def processor():
    return InvoiceProcessor()


@pytest.fixture(scope="module")
def loader():
    return DocumentLoader()


@pytest.fixture(scope="module")
def ocr():
    return OCREngine()


def test_four_sample_invoices_exact_match(processor, loader, ocr):
    ref_path = Path("data/four_sample_invoices/sample_ground_truth_reference.json")
    if not ref_path.exists():
        pytest.skip("four_sample_invoices reference file not found")

    with open(ref_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    for doc_gt in ground_truth:
        filename = doc_gt["file"]
        file_path = Path("data/four_sample_invoices") / filename
        if not file_path.exists():
            continue

        doc = loader.load(file_path)
        tokens = ocr.extract_tokens(doc)
        res = processor.process(doc, tokens)
        fields = res["fields"]

        expected = doc_gt["expected_fields"]

        # 1. Invoice Number: must match and never be "SYNTHETIC"
        assert fields["invoice_number"] == expected["invoice_number"]

        # 2. Supplier Name: must not be "INVOICE" or contain address noise
        assert fields["supplier_name"] == expected["supplier_name"]

        # 3. Buyer Name: must not be duplicated
        assert fields["buyer_name"] == expected["buyer_name"]

        # 4. Financial totals
        assert fields["subtotal"] == float(expected["subtotal"])
        assert fields["tax_amount"] == float(expected["tax_amount"])
        assert fields["total_amount"] == float(expected["total_invoice_value"])

        # 5. Line items count
        assert len(res["line_items"]) == len(doc_gt["line_items"])
