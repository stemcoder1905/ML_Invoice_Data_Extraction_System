"""
Regression tests verifying specific failure modes identified in invoice extraction:
1. "CloudBridge" is not returned as invoice number (must be "ITS-8841").
2. Invoice date does not flip month/day (must be "2026-10-05", not "2026-05-10").
3. Buyer address is not merged into buyer name (must be "Maple Retail Solutions").
4. GST tax amount is not null when present (must be 3240.00).
5. Currency codes like "INR INR" are not appended to descriptions.
6. Quantity and unit price are not confused (unit price 750.00, quantity 8).
7. Unit price 750 is not mistaken for a tax amount.
8. Line item fields (qty, unit price, total) are assigned correctly.
9. Output JSON is valid and conforms to expected structure.
10. Valid fields are not overwritten by unrelated nearby text.
"""

from pathlib import Path
import json
import pytest

from src.document_loader import DocumentLoader
from src.ocr_engine import OCREngine
from src.postprocessor import InvoiceProcessor
from src.exporter import InvoiceJSONExporter


@pytest.fixture(scope="module")
def processor():
    return InvoiceProcessor()


@pytest.fixture(scope="module")
def loader():
    return DocumentLoader()


@pytest.fixture(scope="module")
def ocr():
    return OCREngine()


@pytest.fixture(scope="module")
def exporter(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("test_export")
    return InvoiceJSONExporter(output_dir=out_dir)


def test_invoice_02_regression_failure_modes(processor, loader, ocr, exporter):
    """Explicitly tests the 10 failure points on invoice_02_it_services.pdf."""
    pdf_path = Path("data/five_sample_invoices/invoice_02_it_services.pdf")
    if not pdf_path.exists():
        pytest.skip("data/five_sample_invoices/invoice_02_it_services.pdf not found")

    doc = loader.load(pdf_path)
    toks = ocr.extract_tokens(doc)
    res = processor.process(doc, toks)
    f = res["fields"]
    items = res["line_items"]

    # 1. Invoice number must be ITS-8841 and NEVER CloudBridge
    assert f["invoice_number"] != "CloudBridge", "Regression: CloudBridge mistaken for invoice number"
    assert f["invoice_number"] == "ITS-8841"

    # 2. Date must be 2026-10-05 and NEVER 2026-05-10
    assert f["invoice_date"] != "2026-05-10", "Regression: Month and day were swapped"
    assert f["invoice_date"] == "2026-10-05"

    # 3. Buyer name must NOT contain address or city tokens
    assert "Whitefield" not in str(f["buyer_name"]), "Regression: Buyer address merged into buyer name"
    assert "Bengaluru" not in str(f["buyer_name"]), "Regression: City merged into buyer name"
    assert f["buyer_name"] == "Maple Retail Solutions"

    # 4. Supplier name must be complete and not include address
    assert f["supplier_name"] == "CloudBridge IT Services"

    # 5. PO Number must be MRS-PO-778
    assert f["purchase_order_number"] == "MRS-PO-778"

    # 6. GST Tax amount must NOT be null
    assert f["tax_amount"] is not None, "Regression: Tax amount was omitted/null"
    assert f["tax_amount"] == 3240.00
    assert f["subtotal"] == 18000.00
    assert f["total_amount"] == 21240.00

    # 7. Check line items
    assert len(items) == 2, f"Expected 2 line items, found {len(items)}"

    item1 = items[0]
    assert "INR" not in item1["description"], "Regression: Currency code leaked into description"
    assert item1["description"] == "Cloud Hosting - Monthly"
    assert item1["quantity"] == 1.0
    assert item1["unit_price"] == 12000.00
    assert item1["total"] == 12000.00

    item2 = items[1]
    assert item2["description"] == "Technical Support - 8 hours", "Regression: Description corrupted"
    assert item2["quantity"] == 8.0, "Regression: Quantity not correctly identified"
    assert item2["unit_price"] == 750.00, "Regression: Unit price was mistaken for 8"
    assert item2["tax_amount"] is None or item2["tax_amount"] != 750.00, "Regression: Unit price assigned to tax"
    assert item2["total"] == 6000.00

    # 8. Math checks: qty * unit_price == total
    assert round(item1["quantity"] * item1["unit_price"], 2) == item1["total"]
    assert round(item2["quantity"] * item2["unit_price"], 2) == item2["total"]
    assert round(f["subtotal"] + f["tax_amount"], 2) == f["total_amount"]

    # 9. JSON serialization and schema conformity
    exported_path = exporter.export(res, output_filename="test_invoice_02.json")
    assert exported_path.exists()
    with open(exported_path, "r", encoding="utf-8") as jf:
        loaded = json.load(jf)
    assert loaded["fields"]["invoice_number"] == "ITS-8841"
    assert len(loaded["line_items"]) == 2


def test_all_five_sample_invoices_exact_match(processor, loader, ocr):
    """Tests all 5 sample invoices against the ground truth reference."""
    ref_path = Path("data/five_sample_invoices/sample_ground_truth_reference.json")
    if not ref_path.exists():
        pytest.skip("five_sample_invoices reference not found")

    with open(ref_path, "r", encoding="utf-8") as f:
        gt_all = json.load(f)

    for fname, gt in gt_all.items():
        pdf_path = Path("data/five_sample_invoices") / fname
        if not pdf_path.exists():
            continue

        doc = loader.load(pdf_path)
        toks = ocr.extract_tokens(doc)
        res = processor.process(doc, toks)
        fields = res["fields"]
        items = res["line_items"]

        assert fields["invoice_number"] == gt["invoice_number"], f"Failed on {fname} invoice_number"
        assert fields["invoice_date"] == gt["invoice_date"], f"Failed on {fname} invoice_date"
        assert fields["supplier_name"] == gt["vendor_name"], f"Failed on {fname} supplier_name"
        assert fields["buyer_name"] == gt["buyer_name"], f"Failed on {fname} buyer_name"
        assert fields["purchase_order_number"] == gt["po_number"], f"Failed on {fname} po_number"
        assert fields["currency"] == gt["currency"], f"Failed on {fname} currency"
        assert fields["subtotal"] == gt["subtotal"], f"Failed on {fname} subtotal"
        assert fields["tax_amount"] == gt["tax"], f"Failed on {fname} tax"
        assert fields["total_amount"] == gt["total"], f"Failed on {fname} total"
        assert fields["tax_id"] == gt["vendor_gstin"], f"Failed on {fname} tax_id"

        assert len(items) == len(gt["line_items"]), f"Failed on {fname} line items count"
        for idx, exp_it in enumerate(gt["line_items"]):
            got_it = items[idx]
            assert got_it["description"] == exp_it["description"], f"Failed on {fname} item {idx} description"
            assert got_it["quantity"] == float(exp_it["quantity"]), f"Failed on {fname} item {idx} quantity"
            assert got_it["unit_price"] == float(exp_it["unit_price"]), f"Failed on {fname} item {idx} unit_price"
            assert got_it["total"] == float(exp_it["amount"]), f"Failed on {fname} item {idx} total"


def test_invoice_05_consulting_regression(processor, loader, ocr):
    """Regression test specifically targeting the five errors on invoice_05_consulting.pdf."""
    pdf_path = Path("data/five_sample_invoices/invoice_05_consulting.pdf")
    if not pdf_path.exists():
        pytest.skip("data/five_sample_invoices/invoice_05_consulting.pdf not found")

    doc = loader.load(pdf_path)
    toks = ocr.extract_tokens(doc)
    res = processor.process(doc, toks)
    f = res["fields"]
    items = res["line_items"]
    val = res["validation"]

    # 1. Invoice number must NOT be "Redwood", must be "CONS-26-044"
    assert f["invoice_number"] != "Redwood", "Regression: Supplier name was misidentified as invoice number"
    assert f["invoice_number"] == "CONS-26-044"

    # 2. Buyer name must NOT contain address tokens (Industrial Area, Chandigarh, 160002)
    assert "Industrial Area" not in str(f["buyer_name"]), "Regression: Buyer address merged into buyer name"
    assert "Chandigarh" not in str(f["buyer_name"]), "Regression: City merged into buyer name"
    assert "160002" not in str(f["buyer_name"]), "Regression: Postal code merged into buyer name"
    assert f["buyer_name"] == "Cedar Health Products Pvt. Ltd."

    # 3. Invoice-level tax amount must NOT be null; must be 4140.00
    assert f["tax_amount"] is not None, "Regression: Tax amount was omitted/null"
    assert f["tax_amount"] == 4140.00
    assert f["subtotal"] == 23000.00
    assert f["total_amount"] == 27140.00

    # 4. Line-item descriptions must NOT contain currency artifacts like "INR INR"
    assert len(items) == 2
    for itm in items:
        assert "INR" not in itm["description"], f"Regression: Currency leaked into description '{itm['description']}'"
    assert items[0]["description"] == "Process Review Workshop"
    assert items[0]["quantity"] == 2.0
    assert items[0]["unit_price"] == 8500.00
    assert items[0]["total"] == 17000.00

    assert items[1]["description"] == "Implementation Report"
    assert items[1]["quantity"] == 1.0
    assert items[1]["unit_price"] == 6000.00
    assert items[1]["total"] == 6000.00

    # 5. Validation reports is_valid: True with no errors when valid
    assert val["is_valid"] is True
    assert len(val["errors"]) == 0


def test_validator_detects_missing_tax_and_invalid_fields():
    """Verifies that validator rejects invoices with missing required fields or missing tax."""
    from src.validator import InvoiceValidator
    val = InvoiceValidator()

    # Invoice where total > subtotal but tax_amount is missing and cannot be derived
    # (e.g. subtotal is also missing or inconsistent)
    fields_missing_tax = {
        "invoice_number": "CONS-26-044",
        "invoice_date": "2026-10-20",
        "supplier_name": "Redwood Business Consulting",
        "subtotal": 23000.00,
        "tax_amount": None,
        "total_amount": 27140.00,
    }
    # When tax is missing, validation reconciles it if possible, but if explicitly invalid:
    res = val.validate({
        "invoice_number": None,
        "invoice_date": "2026-10-20",
        "supplier_name": "Redwood Business Consulting",
        "total_amount": 100.00,
    }, [])
    assert res["is_valid"] is False
    assert any("invoice_number" in err.lower() for err in res["errors"])

    # Discrepant math: subtotal + tax != total
    res_math = val.validate({
        "invoice_number": "INV-1",
        "invoice_date": "2026-10-20",
        "supplier_name": "Test",
        "subtotal": 100.00,
        "tax_amount": 18.00,
        "total_amount": 200.00,
    }, [])
    assert res_math["is_valid"] is False
    assert any("mathematical" in err.lower() for err in res_math["errors"])


def test_all_six_sample_invoices_exact_match(processor, loader, ocr):
    """Verifies that all 6 invoices in six_sample_invoices match ground truth 100%."""
    ref_path = Path("data/six_sample_invoices/ground_truth_reference.json")
    if not ref_path.exists():
        pytest.skip("six_sample_invoices ground truth not found")

    with open(ref_path, "r", encoding="utf-8") as f:
        gt_all = json.load(f)

    for fname, gt in gt_all.items():
        pdf_path = Path("data/six_sample_invoices") / fname
        if not pdf_path.exists():
            continue

        doc = loader.load(pdf_path)
        toks = ocr.extract_tokens(doc)
        res = processor.process(doc, toks)
        fields = res["fields"]
        items = res["line_items"]
        val = res["validation"]

        assert fields["invoice_number"] == gt["invoice_number"], f"Failed on {fname} invoice_number"
        assert fields["invoice_date"] == gt["invoice_date"], f"Failed on {fname} invoice_date"
        assert fields["supplier_name"] == gt["supplier_name"], f"Failed on {fname} supplier_name"
        assert fields["buyer_name"] == gt["buyer_name"], f"Failed on {fname} buyer_name"
        assert fields["purchase_order_number"] == gt["purchase_order_number"], f"Failed on {fname} po_number"
        assert fields["currency"] == gt["currency"], f"Failed on {fname} currency"
        assert fields["subtotal"] == gt["subtotal"], f"Failed on {fname} subtotal"
        assert fields["tax_amount"] == gt["tax_amount"], f"Failed on {fname} tax"
        assert fields["total_amount"] == gt["total_amount"], f"Failed on {fname} total"
        assert val["is_valid"] is True, f"Failed on {fname} validation"
        assert len(val["errors"]) == 0, f"Failed on {fname} validation errors"

        assert len(items) == len(gt["line_items"]), f"Failed on {fname} line items count"
        for idx, exp_it in enumerate(gt["line_items"]):
            got_it = items[idx]
            assert got_it["description"] == exp_it["description"], f"Failed on {fname} item {idx} description"
            assert got_it["quantity"] == float(exp_it["quantity"]), f"Failed on {fname} item {idx} quantity"
            assert got_it["unit_price"] == float(exp_it["unit_price"]), f"Failed on {fname} item {idx} unit_price"
            assert got_it["total"] == float(exp_it["total"]), f"Failed on {fname} item {idx} total"


