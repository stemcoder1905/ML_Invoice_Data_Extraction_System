"""
Automated Unit Tests for Candidate Extractor Module.
"""

from src.candidate_extractor import CandidateExtractor


def test_spatial_two_column_header_party_extraction():
    extractor = CandidateExtractor()
    tokens = [
        {"clean_text": "SUPPLIER", "norm_bbox": [0.1, 0.14, 0.2, 0.16], "page_num": 1},
        {"clean_text": "BILL", "norm_bbox": [0.55, 0.14, 0.60, 0.16], "page_num": 1},
        {"clean_text": "TO", "norm_bbox": [0.61, 0.14, 0.65, 0.16], "page_num": 1},
        # Supplier name under SUPPLIER
        {"clean_text": "Blue", "norm_bbox": [0.1, 0.18, 0.15, 0.20], "page_num": 1},
        {"clean_text": "Harbor", "norm_bbox": [0.16, 0.18, 0.22, 0.20], "page_num": 1},
        {"clean_text": "Office", "norm_bbox": [0.23, 0.18, 0.28, 0.20], "page_num": 1},
        {"clean_text": "Supply", "norm_bbox": [0.29, 0.18, 0.35, 0.20], "page_num": 1},
        {"clean_text": "LLC", "norm_bbox": [0.36, 0.18, 0.40, 0.20], "page_num": 1},
        # Buyer name under BILL TO
        {"clean_text": "Cedar", "norm_bbox": [0.55, 0.18, 0.62, 0.20], "page_num": 1},
        {"clean_text": "Technologies", "norm_bbox": [0.63, 0.18, 0.75, 0.20], "page_num": 1},
        {"clean_text": "Inc", "norm_bbox": [0.76, 0.18, 0.80, 0.20], "page_num": 1},
    ]

    parties = extractor._extract_parties_from_tokens(tokens)
    assert parties.get("supplier_name") == "Blue Harbor Office Supply LLC"
    assert parties.get("buyer_name") == "Cedar Technologies Inc"
    assert "Cedar" not in parties.get("supplier_name", "")
    assert parties.get("buyer_name") != "SUPPLIER"


def test_amount_extraction_with_tax_percentage():
    extractor = CandidateExtractor()
    lines = [
        "Subtotal USD 769.92",
        "Tax (8%) USD 61.59",
        "Total Amount Due USD 831.51",
    ]
    amounts = extractor._extract_amounts_from_lines(lines)
    assert amounts.get("subtotal") == "769.92"
    assert amounts.get("tax_amount") == "61.59"
    assert amounts.get("total_amount") == "831.51"
