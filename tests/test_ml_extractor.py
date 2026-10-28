"""
Automated Unit Tests for ML CRF Extractor.
"""

import pytest
from src.ml_extractor import InvoiceCRFExtractor
from src.config import DEFAULT_MODEL_PATH


def test_crf_model_loaded_and_trained():
    extractor = InvoiceCRFExtractor()
    assert extractor.is_trained() is True


def test_crf_prediction_format():
    extractor = InvoiceCRFExtractor()
    sample_tokens = [
        {"clean_text": "TAX", "page_num": 1, "norm_bbox": [0.1, 0.05, 0.2, 0.08]},
        {"clean_text": "INVOICE", "page_num": 1, "norm_bbox": [0.22, 0.05, 0.35, 0.08]},
        {"clean_text": "Invoice", "page_num": 1, "norm_bbox": [0.6, 0.1, 0.7, 0.12]},
        {"clean_text": "No:", "page_num": 1, "norm_bbox": [0.72, 0.1, 0.78, 0.12]},
        {"clean_text": "INV-2024-1001", "page_num": 1, "norm_bbox": [0.8, 0.1, 0.95, 0.12]},
    ]
    preds = extractor.predict_tokens(sample_tokens)
    assert len(preds) == len(sample_tokens)
    assert isinstance(preds[0], str)

    extracted_fields, enriched = extractor.extract_fields(sample_tokens)
    assert isinstance(extracted_fields, dict)
    assert len(enriched) == len(sample_tokens)
    assert "pred_label" in enriched[0]


def test_untrained_fallback_handling(tmp_path):
    empty_model_path = tmp_path / "non_existent_crf.pkl"
    extractor = InvoiceCRFExtractor(model_path=empty_model_path)
    assert extractor.is_trained() is False

    fields, _ = extractor.extract_fields([{"clean_text": "Sample", "page_num": 1, "norm_bbox": [0, 0, 0, 0]}])
    assert fields == {}
