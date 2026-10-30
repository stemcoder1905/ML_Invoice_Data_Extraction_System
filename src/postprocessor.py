"""
Postprocessing and Ensemble Merging Module.
Combines CRF model predictions with candidate rules, merges line items,
applies quality validation, executes multi-document segmentation, and
determines processing provenance.
"""

from typing import Dict, Any, List, Optional
import time
import re

from src.config import TARGET_FIELDS
from src.ml_extractor import InvoiceCRFExtractor
from src.candidate_extractor import CandidateExtractor, deduplicate_phrase, REJECT_PARTY_NAMES, INVALID_INV_NUMS
from src.line_item_extractor import LineItemExtractor
from src.document_segmenter import DocumentSegmenter
from src.validator import InvoiceValidator


class InvoiceProcessor:
    """Master orchestrator combining segmentation, ML inference, candidate rules, table extraction, and validation."""

    def __init__(self, crf_extractor: Optional[InvoiceCRFExtractor] = None):
        self.crf_extractor = crf_extractor or InvoiceCRFExtractor()
        self.candidate_extractor = CandidateExtractor()
        self.line_item_extractor = LineItemExtractor()
        self.segmenter = DocumentSegmenter()
        self.validator = InvoiceValidator()

    def _sanitize_party(self, val: Any) -> Optional[str]:
        if not val:
            return None
        s = str(val).strip()
        s = re.sub(
            r"(?i)^(?:supplier\s+invoice\s+details|supplier|bill\s+to|ship\s+to|billed\s+to|customer\s+name|customer\s+details|customer|buyer|name|legal\s+entity\s+name|restaurant\s+name)\s*[:#\-]?\s*",
            "",
            s,
        ).strip()
        s = deduplicate_phrase(s)
        if s.upper() in REJECT_PARTY_NAMES or len(s) < 2:
            return None
        if re.match(r"^OD\d{10,}", s):
            return None
        if re.match(r"^[\d\s\-\(\)\+]+$", s):
            return None
        if self.candidate_extractor._is_address_line(s):
            return None
        return s

    def _sanitize_inv_num(self, val: Any) -> Optional[str]:
        if not val:
            return None
        s = str(val).strip().strip(":#-.,")
        if s.upper() in INVALID_INV_NUMS:
            return None
        if s.lower().startswith(("synthetic", "sample", "tax", "date", "bill", "invoice")):
            return None
        if s.isalpha() and len(s) > 4:
            return None
        return s

    def process(
        self,
        document_dict: Dict[str, Any],
        tokens: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Executes end-to-end extraction on document tokens supporting both single-document
        and multi-document PDFs with automatic document boundary segmentation.
        """
        start_time = time.time()
        filename = document_dict.get("filename", "unknown")
        doc_type = document_dict.get("doc_type", "unknown")
        page_count = document_dict.get("page_count", 1)

        # 1. Segment pages into distinct financial documents
        segments = self.segmenter.segment_pages(tokens, page_count)

        if len(segments) <= 1:
            # Single document workflow
            res = self._process_single_segment(
                document_dict,
                tokens,
                {"segment_index": 1, "document_type": doc_type, "page_numbers": list(range(1, page_count + 1))},
                start_time=start_time,
            )
            res["multiple_documents_detected"] = False
            res["document_count"] = 1
            res["documents"] = [
                {
                    "document_index": 1,
                    "document_type": res["metadata"].get("document_type", "Invoice"),
                    "page_numbers": list(range(1, page_count + 1)),
                    "fields": res["fields"],
                    "line_items": res["line_items"],
                    "validation": res["validation"],
                }
            ]
            return res
        else:
            # Multi-document PDF workflow (e.g. Flipkart, Amazon, Zomato bundles)
            doc_records = []
            for seg in segments:
                seg_tokens = [t for t in tokens if t.get("page_num", 1) in seg["page_numbers"]]
                seg_res = self._process_single_segment(
                    document_dict, seg_tokens, seg, start_time=start_time
                )
                doc_records.append({
                    "document_index": seg["segment_index"],
                    "document_type": seg.get("document_type", "Invoice"),
                    "page_numbers": seg["page_numbers"],
                    "fields": seg_res["fields"],
                    "line_items": seg_res["line_items"],
                    "validation": seg_res["validation"],
                })

            # Select primary document (highest invoice total or first)
            primary_idx = 0
            max_tot = -1.0
            for idx, d in enumerate(doc_records):
                tot = d["fields"].get("total_amount")
                if tot is not None and tot > max_tot:
                    max_tot = tot
                    primary_idx = idx

            primary_doc = doc_records[primary_idx]
            duration_ms = round((time.time() - start_time) * 1000, 2)
            ocr_engine_used = "native_pdf" if doc_type == "digital_pdf" else "tesseract_ocr"

            # Backward-compatible envelope with all documents preserved
            result = {
                "metadata": {
                    "filename": filename,
                    "document_type": doc_type,
                    "page_count": page_count,
                    "token_count": len(tokens),
                    "ocr_engine": ocr_engine_used,
                    "extraction_method": "multi_document_segmentation",
                    "processing_time_ms": duration_ms,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "multiple_documents_detected": True,
                "document_count": len(doc_records),
                "primary_document_index": primary_doc["document_index"],
                "fields": primary_doc["fields"],
                "line_items": primary_doc["line_items"],
                "validation": primary_doc["validation"],
                "documents": doc_records,
            }
            return result

    def _process_single_segment(
        self,
        document_dict: Dict[str, Any],
        tokens: List[Dict[str, Any]],
        seg_meta: Dict[str, Any],
        start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Processes a single isolated document segment."""
        t_start = start_time or time.time()
        filename = document_dict.get("filename", "unknown")
        doc_type = document_dict.get("doc_type", "unknown")
        pages = seg_meta.get("page_numbers", [1])

        # Build full text and line strings for candidate extractor
        full_text = " ".join(t["clean_text"] for t in tokens)
        token_lines = self.line_item_extractor._group_tokens_by_y(tokens)
        text_lines = [" ".join(tok["clean_text"] for tok in line) for line in token_lines]

        # 1. Deterministic Candidate Extraction
        rule_candidates = self.candidate_extractor.extract_candidates_from_text(
            full_text, text_lines, tokens=tokens
        )

        # 2. Traditional ML CRF Extraction
        crf_fields = {}
        extraction_method = "rule_based_fallback"
        model_trained = self.crf_extractor.is_trained()

        if model_trained and tokens:
            try:
                crf_fields, _ = self.crf_extractor.extract_fields(tokens)
                if crf_fields:
                    extraction_method = "crf_ml"
            except Exception:
                crf_fields = {}

        # 3. Hybrid Merge (Complementary rules for fields CRF missed or where CRF predicted noise)
        final_fields: Dict[str, Any] = {}
        for field in TARGET_FIELDS:
            crf_val = crf_fields.get(field)
            rule_val = rule_candidates.get(field)

            if field in ("supplier_name", "buyer_name"):
                c_clean = self._sanitize_party(crf_val)
                r_clean = self._sanitize_party(rule_val)
                if r_clean and (
                    not c_clean
                    or any(k in str(crf_val).upper() for k in ["INVOICE", "DETAILS", "SUPPLIER", "BILL TO", "SHIP TO"])
                ):
                    final_fields[field] = r_clean
                elif c_clean:
                    final_fields[field] = c_clean
                else:
                    final_fields[field] = r_clean
            elif field == "invoice_number":
                c_inv = self._sanitize_inv_num(crf_val)
                r_inv = self._sanitize_inv_num(rule_val)
                if r_inv is not None:
                    final_fields[field] = r_inv
                elif c_inv is not None:
                    final_fields[field] = c_inv
                else:
                    final_fields[field] = None
            elif field == "tax_id":
                c_tax = str(crf_val).strip() if crf_val is not None else ""
                r_tax = str(rule_val).strip() if rule_val is not None else ""
                is_valid_c = bool(c_tax and "@" not in c_tax and re.match(r"^[A-Z0-9\-\/]{7,25}$", c_tax, re.I))
                is_valid_r = bool(r_tax and "@" not in r_tax and re.match(r"^[A-Z0-9\-\/]{7,25}$", r_tax, re.I))
                if is_valid_r and re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", r_tax):
                    final_fields[field] = r_tax
                elif is_valid_c and re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", c_tax):
                    final_fields[field] = c_tax
                elif is_valid_r:
                    final_fields[field] = r_tax
                elif is_valid_c:
                    final_fields[field] = c_tax
                else:
                    final_fields[field] = None
            elif field == "purchase_order_number":
                c_po = str(crf_val).strip() if crf_val is not None else ""
                r_po = str(rule_val).strip() if rule_val is not None else ""
                is_date_c = bool(re.match(r"^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}$", c_po))
                is_date_r = bool(re.match(r"^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}$", r_po))
                is_valid_c = bool(c_po and not is_date_c and len(c_po) >= 3 and not c_po.lower().startswith(("inv", "tax", "date", "bill")))
                is_valid_r = bool(r_po and not is_date_r and len(r_po) >= 3 and not r_po.lower().startswith(("inv", "tax", "date", "bill")))
                if is_valid_r:
                    final_fields[field] = r_po
                elif is_valid_c:
                    final_fields[field] = c_po
                else:
                    final_fields[field] = None
            elif field in ("subtotal", "tax_amount", "total_amount"):
                def _to_float(v):
                    if v is None:
                        return None
                    try:
                        clean = re.sub(r"[^\d\.]", "", str(v).replace(",", ""))
                        flt = float(clean)
                        return flt if 0.0 <= flt <= 10000000.0 else None
                    except Exception:
                        return None
                c_num = _to_float(crf_val)
                r_num = _to_float(rule_val)
                if r_num is not None:
                    final_fields[field] = r_num
                elif c_num is not None:
                    final_fields[field] = c_num
                else:
                    final_fields[field] = None
            else:
                if crf_val is not None and str(crf_val).strip() != "":
                    final_fields[field] = crf_val
                elif rule_val is not None:
                    final_fields[field] = rule_val
                else:
                    final_fields[field] = None

            if final_fields[field] == rule_val and crf_val != rule_val and model_trained and extraction_method == "crf_ml":
                extraction_method = "hybrid_crf_rules"

        # 4. Table / Line-Item Extraction
        extracted_line_items = self.line_item_extractor.extract_line_items(tokens)

        # 5. Validation and Quality Assurance
        val_result = self.validator.validate(final_fields, extracted_line_items)

        if not model_trained:
            val_result["warnings"].insert(
                0, "CRF model is not trained or loaded. Extraction completed using rule-based candidate fallback."
            )

        duration_ms = round((time.time() - t_start) * 1000, 2)
        ocr_engine_used = "native_pdf" if doc_type == "digital_pdf" else "tesseract_ocr"

        return {
            "metadata": {
                "filename": filename,
                "document_type": seg_meta.get("document_type", doc_type),
                "pages": pages,
                "token_count": len(tokens),
                "ocr_engine": ocr_engine_used,
                "extraction_method": extraction_method,
                "processing_time_ms": duration_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "fields": val_result["cleaned_fields"],
            "line_items": val_result["cleaned_line_items"],
            "validation": {
                "is_valid": val_result["is_valid"],
                "errors": val_result["errors"],
                "warnings": val_result["warnings"],
            },
        }
