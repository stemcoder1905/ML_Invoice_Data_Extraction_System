"""
Automated Unit Tests for Validator, GSTIN, Dates, and Math Consistency.
"""

from src.validator import InvoiceValidator


def test_gstin_validation():
    val = InvoiceValidator()

    # Valid Indian GSTIN (State 27, PAN AABCU9603R, 1 entity, Z, M checksum)
    res_valid = val.validate({"tax_id": "27AABCU9603R1ZM", "total_amount": 100.0, "invoice_number": "INV-1", "invoice_date": "2024-01-01", "supplier_name": "Test"}, [])
    assert res_valid["cleaned_fields"]["tax_id_type"] == "GSTIN"

    # Suspicious short tax ID
    res_short = val.validate({"tax_id": "123", "total_amount": 100.0, "invoice_number": "INV-1", "invoice_date": "2024-01-01", "supplier_name": "Test"}, [])
    assert any("Suspicious Tax ID" in w for w in res_short["warnings"])


def test_date_normalization_and_invalid_date():
    val = InvoiceValidator()

    # Normalization of DMY
    res_date = val.validate({"invoice_date": "15/08/2024", "total_amount": 100.0, "invoice_number": "INV-1", "supplier_name": "Test"}, [])
    assert res_date["cleaned_fields"]["invoice_date"] == "2024-08-15"

    # Completely invalid date
    res_invalid_date = val.validate({"invoice_date": "Not-A-Date-String", "total_amount": 100.0, "invoice_number": "INV-1", "supplier_name": "Test"}, [])
    assert any("Invalid date format" in w for w in res_invalid_date["warnings"])


def test_math_consistency_subtotal_tax_total():
    val = InvoiceValidator()

    # Consistent: 100 + 18 = 118
    fields_ok = {
        "invoice_number": "INV-100",
        "invoice_date": "2024-05-10",
        "supplier_name": "Apex Corp",
        "subtotal": "100.00",
        "tax_amount": "18.00",
        "total_amount": "118.00",
    }
    res_ok = val.validate(fields_ok, [])
    assert not any("Mathematical Discrepancy" in w for w in res_ok["warnings"])

    # Discrepancy: 100 + 18 != 150
    fields_bad = dict(fields_ok)
    fields_bad["total_amount"] = "150.00"
    res_bad = val.validate(fields_bad, [])
    assert any("Mathematical Discrepancy" in w for w in res_bad["warnings"])


def test_missing_fields_warnings():
    val = InvoiceValidator()
    # Completely empty fields
    res = val.validate({}, [])
    assert any("Missing 'invoice_number'" in w for w in res["warnings"])
    assert any("Missing 'invoice_date'" in w for w in res["warnings"])
    assert any("Missing 'total_amount'" in w for w in res["warnings"])
    assert any("Missing 'supplier_name'" in w for w in res["warnings"])


def test_date_normalization_iso_dayfirst_safety():
    val = InvoiceValidator()
    # ISO date 2026-10-09 must remain 2026-10-09 (October 9), NOT 2026-09-10 (September 10)
    norm, err = val._validate_and_normalize_date("2026-10-09")
    assert err is None
    assert norm == "2026-10-09"

    norm2, err2 = val._validate_and_normalize_date("2026-10-01")
    assert err2 is None
    assert norm2 == "2026-10-01"


def test_math_reconciliation_missing_tax_amount():
    val = InvoiceValidator()
    fields = {
        "invoice_number": "INV-100",
        "invoice_date": "2026-10-09",
        "supplier_name": "Blue Harbor LLC",
        "subtotal": 769.92,
        "tax_amount": None,
        "total_amount": 831.51,
    }
    res = val.validate(fields, [])
    assert res["cleaned_fields"]["tax_amount"] == 61.59
    assert res["cleaned_fields"]["subtotal"] == 769.92
    assert res["cleaned_fields"]["total_amount"] == 831.51

