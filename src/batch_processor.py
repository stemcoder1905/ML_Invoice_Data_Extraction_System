"""
Batch Processing Module for Invoice Documents.
Coordinates loading, OCR, ML extraction, validation, and JSON exporting
over collections of PDF and image invoices with progress tracking and fault isolation.
"""

from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
import time
import logging

from src.config import SUPPORTED_EXTENSIONS, OUTPUT_DIR
from src.document_loader import DocumentLoader
from src.ocr_engine import OCREngine
from src.postprocessor import InvoiceProcessor
from src.exporter import InvoiceJSONExporter

logger = logging.getLogger("InvoiceBatchProcessor")


class BatchProcessor:
    """Processes directories or collections of invoice documents in batch."""

    def __init__(
        self,
        processor: Optional[InvoiceProcessor] = None,
        output_dir: Union[str, Path] = OUTPUT_DIR,
    ):
        self.loader = DocumentLoader()
        self.ocr_engine = OCREngine()
        self.processor = processor or InvoiceProcessor()
        self.exporter = InvoiceJSONExporter(output_dir=output_dir)

    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Processes a single invoice file and exports its JSON result."""
        p = Path(file_path)
        try:
            doc_dict = self.loader.load(p)
            tokens = self.ocr_engine.extract_tokens(doc_dict)
            result = self.processor.process(doc_dict, tokens)
            out_json_path = self.exporter.export(result, f"{p.stem}_extracted.json")
            result["metadata"]["output_json_path"] = str(out_json_path)
            return result
        except Exception as e:
            logger.error(f"Error processing {p.name}: {e}")
            error_payload = {
                "metadata": {
                    "filename": p.name,
                    "document_type": "unknown",
                    "page_count": 0,
                    "error": str(e),
                    "extraction_method": "failed",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "fields": {},
                "line_items": [],
                "validation": {
                    "is_valid": False,
                    "errors": [f"Processing failed: {str(e)}"],
                    "warnings": [],
                },
            }
            out_json_path = self.exporter.export(error_payload, f"{p.stem}_error.json")
            error_payload["metadata"]["output_json_path"] = str(out_json_path)
            return error_payload

    def process_batch(
        self,
        files_or_dir: Union[str, Path, List[Union[str, Path]]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Processes multiple invoices with fault tolerance and summary metrics.
        """
        start_time = time.time()
        file_list: List[Path] = []

        if isinstance(files_or_dir, (str, Path)):
            p = Path(files_or_dir)
            if p.is_dir():
                for ext in SUPPORTED_EXTENSIONS:
                    file_list.extend(p.glob(f"*{ext}"))
                    file_list.extend(p.glob(f"*{ext.upper()}"))
            elif p.is_file():
                file_list = [p]
        elif isinstance(files_or_dir, list):
            file_list = [Path(f) for f in files_or_dir]

        total_files = len(file_list)
        results = []
        successful_count = 0
        failed_count = 0
        valid_count = 0

        for idx, file_path in enumerate(file_list):
            if progress_callback:
                progress_callback(idx + 1, total_files, file_path.name)

            res = self.process_file(file_path)
            results.append(res)

            if "error" in res["metadata"]:
                failed_count += 1
            else:
                successful_count += 1
                if res["validation"]["is_valid"]:
                    valid_count += 1

        total_time_sec = round(time.time() - start_time, 2)
        avg_time_ms = round((total_time_sec / max(1, total_files)) * 1000, 2)

        summary = {
            "total_documents": total_files,
            "successful_extractions": successful_count,
            "failed_extractions": failed_count,
            "valid_invoices": valid_count,
            "invalid_or_warning_invoices": total_files - valid_count,
            "total_processing_time_sec": total_time_sec,
            "average_time_per_document_ms": avg_time_ms,
            "results": results,
        }

        return summary
