"""
Automated Unit Tests for Feature Engineering.
"""

from src.feature_engineering import FeatureExtractor


def test_word_shape():
    fe = FeatureExtractor()
    assert fe.word_shape("INV-2024-001") == "XXX-9+-999"
    assert fe.word_shape("Total:") == "Xx+:"
    assert fe.word_shape("2,500.00") == "9,999.99"


def test_token_feature_extraction():
    fe = FeatureExtractor()
    tokens = [
        {"clean_text": "Invoice", "page_num": 1, "norm_bbox": [0.1, 0.1, 0.2, 0.12]},
        {"clean_text": "INV-2024-99", "page_num": 1, "norm_bbox": [0.22, 0.1, 0.4, 0.12]},
        {"clean_text": "Total:", "page_num": 1, "norm_bbox": [0.1, 0.8, 0.2, 0.82]},
        {"clean_text": "1,250.00", "page_num": 1, "norm_bbox": [0.22, 0.8, 0.35, 0.82]},
    ]
    features = fe.extract_sequence_features(tokens)
    assert len(features) == 4

    # First token BOS
    assert features[0].get("BOS") is True
    assert features[0]["kw.invoice"] is True
    assert features[0]["pos.is_top_quarter"] is True

    # Second token has previous keyword invoice
    assert features[1]["-1:kw.invoice"] is True

    # Fourth token has amount pattern and bottom region
    assert features[3]["pat.amount"] is True
    assert features[3]["pos.is_bottom_quarter"] is True
    assert features[3].get("EOS") is True
