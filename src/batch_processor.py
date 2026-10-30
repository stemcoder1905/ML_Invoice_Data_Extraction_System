"""
Batch Processing Module for Invoice Documents.
Coordinates loading, OCR, ML extraction, validation, and JSON exporting
over collections of PDF and image invoices with progress tracking and fault isolation.
"""

from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from pathlib import Path
import time
import logging
import io

from src.config import SUPPORTED_EXTENSIONS, OUTPUT_DIR
from src.document_loader import DocumentLoader
from src.ocr_engine import OCREngine
from src.postprocessor import InvoiceProcessor
from src.exporter import InvoiceJSONExporter

logger = logging.getLogger("InvoiceBatchProcessor")


class BatchProcessor:
    """Processes directories, zip archives, or collections of invoice documents in batch."""

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

    def process_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Processes in-memory invoice bytes and exports its JSON result."""
        stem = Path(filename).stem
        try:
            doc_dict = self.loader.load(file_bytes, filename=filename)
            tokens = self.ocr_engine.extract_tokens(doc_dict)
            result = self.processor.process(doc_dict, tokens)
            out_json_path = self.exporter.export(result, f"{stem}_extracted.json")
            result["metadata"]["output_json_path"] = str(out_json_path)
            return result
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            error_payload = {
                "metadata": {
                    "filename": filename,
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
            out_json_path = self.exporter.export(error_payload, f"{stem}_error.json")
            error_payload["metadata"]["output_json_path"] = str(out_json_path)
            return error_payload

    def process_batch(
        self,
        files_or_dir: Union[str, Path, List[Union[str, Path, Tuple[str, bytes]]]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Processes multiple invoices with fault tolerance, ZIP archive extraction, and summary metrics.
        Accepts:
        - Directory path
        - ZIP file path (.zip)
        - List of file paths / zip paths
        - List of (filename, file_bytes) tuples
        """
        start_time = time.time()
        # Item format: (name, path_or_bytes)
        items_to_process: List[Tuple[str, Union[Path, bytes]]] = []

        if isinstance(files_or_dir, (str, Path)):
            p = Path(files_or_dir)
            if p.is_dir():
                for ext in SUPPORTED_EXTENSIONS:
                    for f in p.glob(f"*{ext}"):
                        items_to_process.append((f.name, f))
                    for f in p.glob(f"*{ext.upper()}"):
                        items_to_process.append((f.name, f))
                for z in p.glob("*.zip"):
                    for z_name, z_bytes in DocumentLoader.extract_zip(z):
                        items_to_process.append((z_name, z_bytes))
            elif p.is_file():
                if p.suffix.lower() == ".zip":
                    for z_name, z_bytes in DocumentLoader.extract_zip(p):
                        items_to_process.append((z_name, z_bytes))
                else:
                    items_to_process.append((p.name, p))
        elif isinstance(files_or_dir, list):
            for item in files_or_dir:
                if isinstance(item, tuple) and len(item) == 2:
                    sub_name, sub_data = item
                    if Path(sub_name).suffix.lower() == ".zip":
                        for z_name, z_bytes in DocumentLoader.extract_zip(sub_data):
                            items_to_process.append((z_name, z_bytes))
                    else:
                        items_to_process.append((sub_name, sub_data))
                else:
                    p = Path(item)
                    if p.suffix.lower() == ".zip":
                        for z_name, z_bytes in DocumentLoader.extract_zip(p):
                            items_to_process.append((z_name, z_bytes))
                    else:
                        items_to_process.append((p.name, p))

        total_files = len(items_to_process)
        results = []
        successful_count = 0
        failed_count = 0
        valid_count = 0

        for idx, (fname, ftarget) in enumerate(items_to_process):
            if progress_callback:
                progress_callback(idx + 1, total_files, fname)

            if isinstance(ftarget, (bytes, bytearray)):
                res = self.process_bytes(bytes(ftarget), filename=fname)
            else:
                res = self.process_file(ftarget)

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
