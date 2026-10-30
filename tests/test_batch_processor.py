"""
Automated Integration Tests for Batch Processing Pipeline.
"""

import io
import zipfile
from pathlib import Path
from src.batch_processor import BatchProcessor
from src.document_loader import DocumentLoader
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


def test_document_loader_extract_zip(tmp_path):
    # Create an in-memory zip archive with 2 PDF invoices and a non-invoice file
    sample_pdfs = list(INVOICES_DIR.glob("*.pdf"))[:2]
    assert len(sample_pdfs) == 2

    zip_path = tmp_path / "test_invoices.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sample_pdfs:
            zf.write(p, arcname=p.name)
        zf.writestr("notes.txt", "Some random notes that should be ignored.")
        zf.writestr("__MACOSX/._invoice.pdf", "junk resource fork")

    extracted = DocumentLoader.extract_zip(zip_path)
    assert len(extracted) == 2
    extracted_names = [e[0] for e in extracted]
    assert sample_pdfs[0].name in extracted_names
    assert sample_pdfs[1].name in extracted_names


def test_batch_processor_zip_archive(tmp_path):
    # Test batch processing directly with a ZIP archive
    sample_pdfs = list(INVOICES_DIR.glob("*.pdf"))[:2]
    zip_path = tmp_path / "batch_invoices.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sample_pdfs:
            zf.write(p, arcname=p.name)

    processor = BatchProcessor(output_dir=tmp_path / "outputs")
    summary = processor.process_batch(zip_path)

    assert summary["total_documents"] == 2
    assert summary["successful_extractions"] == 2
    assert summary["failed_extractions"] == 0
    assert len(summary["results"]) == 2

