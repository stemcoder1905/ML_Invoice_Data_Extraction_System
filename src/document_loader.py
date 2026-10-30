"""
Document Loader Module for Invoices.
Handles PDF (digital and scanned) and image formats (PNG, JPG, JPEG).
Renders pages, extracts native PDF text when available, and prepares pages for OCR.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, BinaryIO, Tuple
import numpy as np
import cv2
import pymupdf as fitz  # PyMuPDF
from PIL import Image
import io
import zipfile

from src.config import SUPPORTED_EXTENSIONS, ALL_SUPPORTED_EXTENSIONS


class DocumentLoader:
    """Loads and standardizes input documents into structured page representations."""

    def __init__(self, render_dpi: int = 300):
        self.render_dpi = render_dpi

    @staticmethod
    def extract_zip(source: Union[str, Path, bytes, BinaryIO]) -> List[Tuple[str, bytes]]:
        """
        Extracts supported invoice files (.pdf, .png, .jpg, .jpeg) from a ZIP archive.
        Returns a list of (filename, file_bytes).
        """
        data: bytes
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"ZIP file not found: {source}")
            with open(p, "rb") as f:
                data = f.read()
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        elif hasattr(source, "read"):
            data = source.read()
        else:
            raise ValueError(f"Unsupported source type for ZIP extraction: {type(source)}")

        extracted_files: List[Tuple[str, bytes]] = []
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                name = member.filename
                # Skip macOS metadata or hidden files
                if "__MACOSX" in name or Path(name).name.startswith("."):
                    continue
                ext = Path(name).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    file_bytes = zf.read(member.filename)
                    clean_name = Path(name).name
                    extracted_files.append((clean_name, file_bytes))

        return extracted_files

    def load(self, source: Union[str, Path, bytes, BinaryIO], filename: str = "document") -> Dict[str, Any]:
        """
        Loads a document from file path, raw bytes, or stream.
        Supports PDF, PNG, JPG, JPEG, and extracts the primary invoice if a ZIP archive is provided.
        Returns a document dictionary with metadata and extracted page objects.
        """
        ext = ""
        data_bytes = None

        if isinstance(source, (str, Path)):
            p = Path(source)
            filename = p.name
            ext = p.suffix.lower()
            if not p.exists():
                raise FileNotFoundError(f"Input file not found: {source}")
            with open(p, "rb") as f:
                data_bytes = f.read()
        elif isinstance(source, (bytes, bytearray)):
            data_bytes = bytes(source)
            ext = Path(filename).suffix.lower()
        elif hasattr(source, "read"):
            data_bytes = source.read()
            ext = Path(filename).suffix.lower()
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Handle ZIP archive input directly
        if ext == ".zip" or (data_bytes and data_bytes.startswith(b"PK\x03\x04")):
            extracted = self.extract_zip(data_bytes)
            if not extracted:
                raise ValueError(f"ZIP archive '{filename}' contains no supported invoice files ({SUPPORTED_EXTENSIONS}).")
            primary_name, primary_bytes = extracted[0]
            loaded_doc = self.load(primary_bytes, filename=primary_name)
            loaded_doc["zip_archive"] = filename
            loaded_doc["zip_total_invoices"] = len(extracted)
            return loaded_doc

        if ext not in SUPPORTED_EXTENSIONS and ext != "":
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {ALL_SUPPORTED_EXTENSIONS}")

        # Distinguish between PDF and image
        if ext == ".pdf" or data_bytes.startswith(b"%PDF"):
            return self._load_pdf(data_bytes, filename)
        else:
            return self._load_image(data_bytes, filename)

    def _load_pdf(self, data: bytes, filename: str) -> Dict[str, Any]:
        """Loads a PDF document and evaluates digital text vs scanned pages."""
        doc = fitz.open(stream=data, filetype="pdf")
        pages: List[Dict[str, Any]] = []
        total_native_words = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            rect = page.rect
            page_w, page_h = rect.width, rect.height

            # Extract native text tokens directly from PyMuPDF
            # Each word tuple: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            words = page.get_text("words")
            total_native_words += len(words)

            # Render page to high-res image (RGB numpy array) for OCR or visual rendering
            pix = page.get_pixmap(dpi=self.render_dpi)
            img_bytes = pix.tobytes("png")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            page_dict = {
                "page_number": page_idx + 1,
                "width": page_w,
                "height": page_h,
                "image": img_bgr,
                "image_width": pix.width,
                "image_height": pix.height,
                "native_words": words,
                "has_native_text": len(words) > 10,
            }
            pages.append(page_dict)

        doc_type = "digital_pdf" if total_native_words >= 15 else "scanned_pdf"

        return {
            "filename": filename,
            "doc_type": doc_type,
            "page_count": len(pages),
            "pages": pages,
        }

    def _load_image(self, data: bytes, filename: str) -> Dict[str, Any]:
        """Loads a raster image file (PNG, JPG, JPEG)."""
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback to PIL decode
            pil_img = Image.open(io.BytesIO(data))
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w = img_bgr.shape[:2]
        page_dict = {
            "page_number": 1,
            "width": float(w),
            "height": float(h),
            "image": img_bgr,
            "image_width": w,
            "image_height": h,
            "native_words": [],
            "has_native_text": False,
        }

        return {
            "filename": filename,
            "doc_type": "image",
            "page_count": 1,
            "pages": [page_dict],
        }
