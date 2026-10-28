"""
Automated Unit Tests for Line Item Table Extraction.
"""

from src.line_item_extractor import LineItemExtractor


def test_line_item_table_detection():
    extractor = LineItemExtractor()

    # Construct synthetic table tokens
    tokens = [
        # Table Header
        {"clean_text": "Description", "norm_bbox": [0.1, 0.4, 0.3, 0.42], "page_num": 1},
        {"clean_text": "Quantity", "norm_bbox": [0.45, 0.4, 0.55, 0.42], "page_num": 1},
        {"clean_text": "Rate", "norm_bbox": [0.65, 0.4, 0.75, 0.42], "page_num": 1},
        {"clean_text": "Amount", "norm_bbox": [0.85, 0.4, 0.95, 0.42], "page_num": 1},
        # Item 1
        {"clean_text": "Cloud", "norm_bbox": [0.1, 0.45, 0.18, 0.47], "page_num": 1},
        {"clean_text": "Hosting", "norm_bbox": [0.19, 0.45, 0.28, 0.47], "page_num": 1},
        {"clean_text": "2", "norm_bbox": [0.48, 0.45, 0.52, 0.47], "page_num": 1},
        {"clean_text": "150.00", "norm_bbox": [0.68, 0.45, 0.74, 0.47], "page_num": 1},
        {"clean_text": "300.00", "norm_bbox": [0.88, 0.45, 0.94, 0.47], "page_num": 1},
        # Item 2
        {"clean_text": "Domain", "norm_bbox": [0.1, 0.50, 0.18, 0.52], "page_num": 1},
        {"clean_text": "Registration", "norm_bbox": [0.19, 0.50, 0.32, 0.52], "page_num": 1},
        {"clean_text": "1", "norm_bbox": [0.48, 0.50, 0.52, 0.52], "page_num": 1},
        {"clean_text": "25.00", "norm_bbox": [0.68, 0.50, 0.74, 0.52], "page_num": 1},
        {"clean_text": "25.00", "norm_bbox": [0.88, 0.50, 0.94, 0.52], "page_num": 1},
        # Table Footer
        {"clean_text": "Subtotal:", "norm_bbox": [0.65, 0.60, 0.75, 0.62], "page_num": 1},
        {"clean_text": "325.00", "norm_bbox": [0.85, 0.60, 0.95, 0.62], "page_num": 1},
    ]

    items = extractor.extract_line_items(tokens)
    assert len(items) == 2

    assert "Cloud Hosting" in items[0]["description"]
    assert items[0]["quantity"] == 2.0
    assert items[0]["unit_price"] == 150.0
    assert items[0]["total"] == 300.0

    assert "Domain Registration" in items[1]["description"]
    assert items[1]["quantity"] == 1.0
    assert items[1]["unit_price"] == 25.0
    assert items[1]["total"] == 25.0
