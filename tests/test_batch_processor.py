"""
Automated Integration Tests for Batch Processing Pipeline.
"""

from pathlib import Path
from src.batch_processor import BatchProcessor
from src.config import INVOICES_DIR


def test_batch_processor_multiple_files(tmp_path):
    processor = BatchProcessor(output_dir=tmp_path)
    pdf_files = list(INVOICES_DIR.glob("*.pdf"))[:3]
    assert len(pdf_files) >= 2

    progress_log = []

    def on_progress(current, total, filename):
        progress_log.append((current, total, filename))

    summary = processor.process_batch(pdf_files, progress_callback=on_progress)

    assert summary["total_documents"] == len(pdf_files)
    assert summary["successful_extractions"] == len(pdf_files)
    assert summary["failed_extractions"] == 0
    assert len(progress_log) == len(pdf_files)
    assert len(summary["results"]) == len(pdf_files)

    for res in summary["results"]:
        assert "metadata" in res
        assert "fields" in res
        assert "line_items" in res
        assert "validation" in res
        out_path = Path(res["metadata"]["output_json_path"])
        assert out_path.exists()


def test_batch_processor_corrupted_file_fault_tolerance(tmp_path):
    processor = BatchProcessor(output_dir=tmp_path)
    corrupted_file = tmp_path / "corrupted_invoice.pdf"
    corrupted_file.write_bytes(b"NOT_A_VALID_PDF_HEADER_DATA")

    res = processor.process_file(corrupted_file)
    # The batch processor must not crash, and must report the error cleanly
    assert "error" in res["metadata"]
    assert res["validation"]["is_valid"] is False
    assert len(res["validation"]["errors"]) > 0
