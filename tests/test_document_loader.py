"""
Automated Unit and Integration Tests for Document Loading.
"""

import pytest
from pathlib import Path
import numpy as np
import cv2

from src.document_loader import DocumentLoader
from src.config import INVOICES_DIR


def test_load_digital_pdf():
    loader = DocumentLoader()
    pdf_files = list(INVOICES_DIR.glob("*.pdf"))
    assert len(pdf_files) > 0, "No sample PDF invoices found."

    doc_dict = loader.load(pdf_files[0])
    assert doc_dict["doc_type"] == "digital_pdf"
    assert doc_dict["page_count"] >= 1
    assert len(doc_dict["pages"]) >= 1

    first_page = doc_dict["pages"][0]
    assert first_page["has_native_text"] is True
    assert len(first_page["native_words"]) > 0
    assert first_page["image"] is not None
    assert isinstance(first_page["image"], np.ndarray)


def test_load_image(tmp_path):
    loader = DocumentLoader()
    # Create a synthetic test image
    img = np.ones((400, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST INVOICE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    img_path = tmp_path / "test_invoice.png"
    cv2.imwrite(str(img_path), img)

    doc_dict = loader.load(img_path)
    assert doc_dict["doc_type"] == "image"
    assert doc_dict["page_count"] == 1
    assert doc_dict["pages"][0]["has_native_text"] is False
    assert doc_dict["pages"][0]["image"].shape[0] == 400


def test_load_non_existent_file():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file_12345.pdf")


def test_load_unsupported_format(tmp_path):
    loader = DocumentLoader()
    bad_file = tmp_path / "document.docx"
    bad_file.write_text("dummy text")
    with pytest.raises(ValueError) as excinfo:
        loader.load(bad_file)
    assert "Unsupported file format" in str(excinfo.value)
