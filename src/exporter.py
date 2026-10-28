"""
JSON Exporter Module for Extracted Invoice Payloads.
Ensures valid JSON serialization, safe handling of missing values,
NumPy/Decimal type conversions, and file output formatting.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import json
import math
from datetime import date, datetime
import numpy as np

from src.config import OUTPUT_DIR


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy types, dates, and non-standard objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def sanitize_for_json(data: Any) -> Any:
    """Recursively converts NaN or Infinite values to None for RFC 8259 JSON compliance."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data


class InvoiceJSONExporter:
    """Serializes invoice extraction results into compliant JSON files."""

    def __init__(self, output_dir: Union[str, Path] = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, payload: Dict[str, Any], output_filename: Optional[str] = None) -> Path:
        """
        Saves extraction payload to a JSON file.
        Returns the Path to the saved file.
        """
        if not output_filename:
            orig_name = payload.get("metadata", {}).get("filename", "invoice")
            stem = Path(orig_name).stem
            output_filename = f"{stem}_extracted.json"

        sanitized_payload = sanitize_for_json(payload)
        out_path = self.output_dir / output_filename

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sanitized_payload, f, indent=2, cls=CustomJSONEncoder, ensure_ascii=False)

        return out_path

    def to_json_string(self, payload: Dict[str, Any]) -> str:
        """Serializes extraction payload directly to a JSON string."""
        sanitized_payload = sanitize_for_json(payload)
        return json.dumps(sanitized_payload, indent=2, cls=CustomJSONEncoder, ensure_ascii=False)
