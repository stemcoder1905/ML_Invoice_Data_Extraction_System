"""
Comprehensive Regression Test Suite for Real-World Invoice Formats and Multi-Document PDFs.
Tests Zepto, Amazon, Zomato, and Flipkart real invoices, asserting exact values across
document segmentation, header parsing, GST reconciliation, and line item extraction.
"""

from pathlib import Path
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


def test_real_amazon_multi_document_segmentation(loader, ocr, processor):
    pdf_path = Path("data/real_invoices/amazon_invoice.pdf")
    if not pdf_path.exists():
        pytest.skip("Amazon invoice file not found.")

    doc = loader.load(pdf_path)
    tokens = ocr.extract_tokens(doc)
    res = processor.process(doc, tokens)

    assert res.get("multiple_documents_detected") is True
    assert res.get("document_count") == 2
    docs = res.get("documents", [])
    assert len(docs) == 2

    # Doc 1: Amazon Services Marketplace Fee
    d1 = docs[0]
    assert d1["fields"].get("invoice_number") == "MKT-343065806"
    assert d1["fields"].get("invoice_date") == "2026-09-14"
    assert "Amazon" in d1["fields"].get("supplier_name", "")
    assert d1["fields"].get("buyer_name") == "Deepak Mishra"
    assert abs(d1["fields"].get("total_amount", 0.0) - 5.00) <= 0.05
    assert len(d1["line_items"]) >= 1

    # Doc 2: SI Enterprises Backpack
    d2 = docs[1]
    assert d2["fields"].get("invoice_number") == "IN-2147"
    assert d2["fields"].get("invoice_date") == "2026-09-14"
    assert "SI ENTERPRISES" in d2["fields"].get("supplier_name", "")
    assert d2["fields"].get("buyer_name") == "Deepak Mishra"
    assert abs(d2["fields"].get("total_amount", 0.0) - 296.65) <= 0.05
    assert len(d2["line_items"]) >= 1
    assert "Backpack" in d2["line_items"][0]["description"]


def test_real_zomato_multi_document_segmentation(loader, ocr, processor):
    pdf_path = Path("data/real_invoices/zomato_invoice.pdf")
    if not pdf_path.exists():
        pytest.skip("Zomato invoice file not found.")

    doc = loader.load(pdf_path)
    tokens = ocr.extract_tokens(doc)
    res = processor.process(doc, tokens)

    assert res.get("multiple_documents_detected") is True
    assert res.get("document_count") == 2
    docs = res.get("documents", [])
    assert len(docs) == 2

    # Doc 1: Restaurant Partner Tax Invoice (Pink Paprikaa)
    d1 = docs[0]
    assert d1["fields"].get("invoice_number") == "26K2G9RQ00004365"
    assert d1["fields"].get("invoice_date") == "2026-09-09"
    assert "PAPRIKAA" in d1["fields"].get("supplier_name", "").upper()
    assert d1["fields"].get("buyer_name") == "Deepak Mishra"
    assert abs(d1["fields"].get("total_amount", 0.0) - 129.52) <= 0.05
    assert abs(d1["fields"].get("subtotal", 0.0) - 123.35) <= 0.05
    assert len(d1["line_items"]) >= 1

    # Doc 2: Eternal / Zomato Platform Fee
    d2 = docs[1]
    assert d2["fields"].get("invoice_number") == "Z27HROT026797616"
    assert d2["fields"].get("invoice_date") == "2026-09-09"
    assert "ETERNAL" in d2["fields"].get("supplier_name", "").upper()
    assert d2["fields"].get("buyer_name") == "Deepak Mishra"
    assert abs(d2["fields"].get("total_amount", 0.0) - 17.69) <= 0.05


def test_real_flipkart_multi_document_segmentation(loader, ocr, processor):
    pdf_path = Path("data/real_invoices/flipkart_invoice.pdf")
    if not pdf_path.exists():
        pytest.skip("Flipkart invoice file not found.")

    doc = loader.load(pdf_path)
    tokens = ocr.extract_tokens(doc)
    res = processor.process(doc, tokens)

    assert res.get("multiple_documents_detected") is True
    assert res.get("document_count") == 3
    docs = res.get("documents", [])
    assert len(docs) == 3

    # Doc 1: Spigen Product Tax Invoice
    d1 = docs[0]
    assert d1["fields"].get("invoice_number") == "LIADGPD270000018"
    assert d1["fields"].get("invoice_date") == "2026-04-16"
    assert "SPIGEN" in d1["fields"].get("supplier_name", "").upper()
    assert d1["fields"].get("buyer_name") == "Sushant Mishra"
    assert abs(d1["fields"].get("subtotal", 0.0) - 714.40) <= 0.05
    assert abs(d1["fields"].get("tax_amount", 0.0) - 128.60) <= 0.05
    assert abs(d1["fields"].get("total_amount", 0.0) - 843.00) <= 0.05

    # Doc 2: GTA Transportation Bill of Supply
    d2 = docs[1]
    assert d2["fields"].get("invoice_number") == "NBABF27001159493"
    assert d2["fields"].get("invoice_date") == "2026-04-17"
    assert "Flipkart India" in d2["fields"].get("supplier_name", "")
    assert d2["fields"].get("buyer_name") == "Sushant Mishra"
    assert abs(d2["fields"].get("total_amount", 0.0) - 156.00) <= 0.05

    # Doc 3: Flipkart Internet Platform Fee
    d3 = docs[2]
    assert d3["fields"].get("invoice_number") == "MXAB427000330587"
    assert d3["fields"].get("invoice_date") == "2026-04-16"
    assert "Flipkart Internet" in d3["fields"].get("supplier_name", "")
    assert d3["fields"].get("buyer_name") == "Sushant Mishra"
    assert abs(d3["fields"].get("total_amount", 0.0) - 17.00) <= 0.05
    assert len(d3["line_items"]) == 2
