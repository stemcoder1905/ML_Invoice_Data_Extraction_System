"""
OCR and Text Extraction Engine.
Extracts token-level bounding boxes and text from digital PDFs via PyMuPDF
and from scanned pages/images via Tesseract OCR (pytesseract) with image preprocessing.
"""

from typing import List, Dict, Any, Optional
import os
import pytesseract
from pytesseract import Output
import numpy as np

from src.config import TESSERACT_CMD
from src.image_preprocessing import ImagePreprocessor


class OCREngine:
    """Extracts words, bounding boxes, lines, and layout metadata from document pages."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd or TESSERACT_CMD
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        self.preprocessor = ImagePreprocessor()
        self.tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        """Verifies if Tesseract executable is functional."""
        try:
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_tokens(self, document_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts words and positional coordinates across all pages of a loaded document.
        Uses native PDF text when available (exact, noiseless), or runs Tesseract on preprocessed images.
        """
        all_tokens: List[Dict[str, Any]] = []

        for page in document_dict.get("pages", []):
            page_num = page.get("page_number", 1)
            img = page.get("image")
            img_h = img.shape[0] if img is not None else 1000.0
            img_w = img.shape[1] if img is not None else 800.0
            page_w = float(page.get("width", img_w))
            page_h = float(page.get("height", img_h))

            # 1. Prefer native digital text when present
            if page.get("has_native_text", False) and page.get("native_words"):
                page_tokens = self._extract_from_native_words(
                    page["native_words"], page_num, page_w, page_h
                )
                all_tokens.extend(page_tokens)
            else:
                # 2. Use Tesseract OCR on preprocessed page image
                img = page.get("image")
                if img is None:
                    continue
                page_tokens = self._extract_from_ocr(img, page_num)
                all_tokens.extend(page_tokens)

        # Preserve natural reading order from PyMuPDF/OCR engine, ordering primarily by page
        all_tokens.sort(key=lambda t: t.get("page_num", 1))
        return all_tokens

    def _extract_from_native_words(
        self, words: List[Any], page_num: int, page_w: float, page_h: float
    ) -> List[Dict[str, Any]]:
        """Parses PyMuPDF native word tuples: (x0, y0, x1, y1, "word", block_no, line_no, word_no)."""
        tokens: List[Dict[str, Any]] = []
        for w_tuple in words:
            if len(w_tuple) < 8:
                continue
            x0, y0, x1, y1, text, block_num, line_num, word_num = w_tuple[:8]
            clean = text.strip()
            if not clean:
                continue

            # Ensure valid non-zero page boundaries
            pw = max(page_w, 1.0)
            ph = max(page_h, 1.0)

            norm_bbox = [
                max(0.0, min(1.0, x0 / pw)),
                max(0.0, min(1.0, y0 / ph)),
                max(0.0, min(1.0, x1 / pw)),
                max(0.0, min(1.0, y1 / ph)),
            ]

            token = {
                "text": text,
                "clean_text": clean,
                "page_num": page_num,
                "block_num": int(block_num),
                "line_num": int(line_num),
                "word_num": int(word_num),
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "norm_bbox": norm_bbox,
                "confidence": 100.0,
                "source": "native_pdf",
            }
            tokens.append(token)
        return tokens

    def _extract_from_ocr(self, image: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """Runs image preprocessing and Tesseract OCR with word-level bounding boxes."""
        if not self.tesseract_available:
            # Re-check in case Tesseract was configured dynamically
            self.tesseract_available = self._check_tesseract()
            if not self.tesseract_available:
                raise RuntimeError(
                    "Tesseract OCR executable not detected or not found in system PATH. "
                    "For scanned images or scanned PDFs, please install Tesseract OCR "
                    "(e.g., 'winget install UB-Mannheim.TesseractOCR') or set TESSERACT_CMD."
                )

        # Preprocess image with OpenCV (resize, deskew, noise filter, binarize)
        processed_img, _ = self.preprocessor.preprocess(image)
        h, w = processed_img.shape[:2]

        # Extract OCR data using pytesseract
        ocr_data = pytesseract.image_to_data(
            processed_img,
            output_type=Output.DICT,
            config="--oem 3 --psm 6",
        )

        tokens: List[Dict[str, Any]] = []
        n_boxes = len(ocr_data["text"])

        for i in range(n_boxes):
            text = ocr_data["text"][i]
            clean = text.strip()
            if not clean:
                continue

            conf = float(ocr_data["conf"][i])
            if conf < 10.0:  # Ignore extreme noise artifacts
                continue

            x = float(ocr_data["left"][i])
            y = float(ocr_data["top"][i])
            bw = float(ocr_data["width"][i])
            bh = float(ocr_data["height"][i])

            norm_bbox = [
                max(0.0, min(1.0, x / w)),
                max(0.0, min(1.0, y / h)),
                max(0.0, min(1.0, (x + bw) / w)),
                max(0.0, min(1.0, (y + bh) / h)),
            ]

            token = {
                "text": text,
                "clean_text": clean,
                "page_num": page_num,
                "block_num": int(ocr_data["block_num"][i]),
                "line_num": int(ocr_data["line_num"][i]),
                "word_num": int(ocr_data["word_num"][i]),
                "bbox": [x, y, x + bw, y + bh],
                "norm_bbox": norm_bbox,
                "confidence": conf,
                "source": "tesseract_ocr",
            }
            tokens.append(token)

        return tokens

    def group_tokens_into_lines(self, tokens: List[Dict[str, Any]], y_tolerance: float = 0.008) -> List[List[Dict[str, Any]]]:
        """
        Groups tokens into visual text lines based on page number and normalized vertical alignment.
        """
        if not tokens:
            return []

        # Sort by page, then y0
        sorted_tokens = sorted(tokens, key=lambda t: (t["page_num"], t["norm_bbox"][1], t["norm_bbox"][0]))
        lines: List[List[Dict[str, Any]]] = []
        current_line: List[Dict[str, Any]] = []
        current_page = None
        current_y = None

        for t in sorted_tokens:
            p = t["page_num"]
            y = t["norm_bbox"][1]

            if current_page is None:
                current_page = p
                current_y = y
                current_line = [t]
            elif p != current_page or abs(y - current_y) > y_tolerance:
                if current_line:
                    # Sort words horizontally
                    current_line.sort(key=lambda tok: tok["norm_bbox"][0])
                    lines.append(current_line)
                current_page = p
                current_y = y
                current_line = [t]
            else:
                current_line.append(t)

        if current_line:
            current_line.sort(key=lambda tok: tok["norm_bbox"][0])
            lines.append(current_line)

        return lines
