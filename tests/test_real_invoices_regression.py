"""
Regression Test Suite for Real-World Invoice Formats.
Tests production invoices across e-commerce, grocery quick-commerce (Zepto, Blinkit, Instamart),
and multi-column Indian GST bills of supply.
"""

from pathlib import Path
import pytest
from src.document_loader import DocumentLoader
from src.ocr_engine import OCREngine
from src.postprocessor import InvoiceProcessor


def test_zepto_real_invoice_extraction():
    """Validates complete header, tax reconciliation, and multi-line line-item extraction on Zepto invoice."""
    pdf_path = Path("data/real_invoices/zepto_invoice_01.pdf")
    if not pdf_path.exists():
        pytest.skip("Real invoice file not found.")

    loader = DocumentLoader()
    ocr_engine = OCREngine()
    processor = InvoiceProcessor()

    doc_dict = loader.load(pdf_path)
    tokens = ocr_engine.extract_tokens(doc_dict)
    result = processor.process(doc_dict, tokens)

    fields = result["fields"]
    line_items = result["line_items"]
    validation = result["validation"]

    # 1. Header & Financial Fields
    assert fields.get("invoice_number") == "26096C0002839095"
    assert fields.get("invoice_date") == "2026-09-17"
    assert fields.get("supplier_name") == "Commodum Groceries Private Limited"
    assert fields.get("buyer_name") == "Sushant mishra"
    assert fields.get("purchase_order_number") == "MLIMKGGPR62331A"
    assert fields.get("currency") == "INR"
    assert fields.get("tax_id") == "06AAKCC1645G1ZT"
    assert fields.get("tax_id_type") == "GSTIN"
    assert abs(fields.get("subtotal", 0.0) - 191.43) <= 0.05
    assert abs(fields.get("tax_amount", 0.0) - 9.58) <= 0.05
    assert abs(fields.get("total_amount", 0.0) - 201.01) <= 0.05

    # 2. Line Items (3 distinct items extracted, no Page 2 contamination)
    assert len(line_items) == 3

    item1 = line_items[0]
    assert "Gud Nariyal Barfi" in item1["description"]
    assert "Bansiwala" in item1["description"]
    assert item1["quantity"] == 1.0
    assert abs(item1["taxable_amount"] - 97.14) <= 0.05
    assert abs(item1["total"] - 102.00) <= 0.05

    item2 = line_items[1]
    assert "Nestle MUNCH MAX" in item2["description"]
    assert item2["quantity"] == 1.0
    assert abs(item2["taxable_amount"] - 23.81) <= 0.05
    assert abs(item2["total"] - 25.00) <= 0.05

    item3 = line_items[2]
    assert "Roasted Chana" in item3["description"]
    assert item3["quantity"] == 1.0
    assert abs(item3["taxable_amount"] - 70.48) <= 0.05
    assert abs(item3["total"] - 74.00) <= 0.05

    # 3. Mathematical Validation
    assert validation["is_valid"] is True
    assert len(validation["errors"]) == 0
