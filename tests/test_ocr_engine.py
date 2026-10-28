"""
Automated Unit Tests for OCR Engine and Token Extraction.
"""

import pytest
import numpy as np

from src.ocr_engine import OCREngine
from src.document_loader import DocumentLoader
from src.config import INVOICES_DIR


def test_extract_tokens_native_pdf():
    ocr_engine = OCREngine()
    loader = DocumentLoader()
    pdf_files = list(INVOICES_DIR.glob("*.pdf"))
    assert len(pdf_files) > 0

    doc_dict = loader.load(pdf_files[0])
    tokens = ocr_engine.extract_tokens(doc_dict)

    assert len(tokens) > 20
    first_tok = tokens[0]
    assert "text" in first_tok
    assert "clean_text" in first_tok
    assert "norm_bbox" in first_tok
    assert len(first_tok["norm_bbox"]) == 4

    # Normalized bounding box coordinates must be strictly within [0.0, 1.0]
    for t in tokens:
        for coord in t["norm_bbox"]:
            assert 0.0 <= coord <= 1.0


def test_group_tokens_into_lines():
    ocr_engine = OCREngine()
    tokens = [
        {"clean_text": "Invoice", "page_num": 1, "norm_bbox": [0.1, 0.2, 0.2, 0.22]},
        {"clean_text": "No:", "page_num": 1, "norm_bbox": [0.22, 0.2, 0.28, 0.22]},
        {"clean_text": "1001", "page_num": 1, "norm_bbox": [0.3, 0.2, 0.38, 0.22]},
        {"clean_text": "Date:", "page_num": 1, "norm_bbox": [0.1, 0.3, 0.2, 0.32]},
        {"clean_text": "2024-01-01", "page_num": 1, "norm_bbox": [0.22, 0.3, 0.35, 0.32]},
    ]
    lines = ocr_engine.group_tokens_into_lines(tokens)
    assert len(lines) == 2
    assert [t["clean_text"] for t in lines[0]] == ["Invoice", "No:", "1001"]
    assert [t["clean_text"] for t in lines[1]] == ["Date:", "2024-01-01"]


def test_ocr_failure_handling_on_unavailable_tesseract():
    # Simulate unavailable tesseract engine
    engine = OCREngine(tesseract_cmd="invalid_tesseract_path_bin")
    engine.tesseract_available = False

    dummy_doc = {
        "doc_type": "image",
        "pages": [{"page_number": 1, "image": np.zeros((100, 100, 3), dtype=np.uint8), "has_native_text": False}],
    }
    with pytest.raises(RuntimeError) as excinfo:
        engine.extract_tokens(dummy_doc)
    assert "Tesseract OCR executable not detected" in str(excinfo.value)
