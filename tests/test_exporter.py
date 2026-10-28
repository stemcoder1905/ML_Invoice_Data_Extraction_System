"""
Automated Unit Tests for JSON Exporter and Safe Serialization.
"""

import json
import numpy as np

from src.exporter import InvoiceJSONExporter, sanitize_for_json


def test_sanitize_and_numpy_json_serialization(tmp_path):
    exporter = InvoiceJSONExporter(output_dir=tmp_path)

    payload = {
        "metadata": {"filename": "test_inv.pdf"},
        "fields": {
            "invoice_number": "INV-999",
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(128.50),
            "numpy_nan": np.float64(np.nan),
            "regular_nan": float("nan"),
            "regular_inf": float("inf"),
        },
        "line_items": [
            {"description": "Item 1", "total": np.float64(50.0)},
        ],
    }

    # Serialization should not crash on numpy types or NaNs
    json_str = exporter.to_json_string(payload)
    parsed = json.loads(json_str)

    assert parsed["fields"]["numpy_int"] == 42
    assert parsed["fields"]["numpy_float"] == 128.50
    assert parsed["fields"]["numpy_nan"] is None
    assert parsed["fields"]["regular_nan"] is None
    assert parsed["fields"]["regular_inf"] is None

    out_file = exporter.export(payload, "test_output.json")
    assert out_file.exists()
