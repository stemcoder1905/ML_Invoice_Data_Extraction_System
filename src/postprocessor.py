"""
Postprocessing and Ensemble Merging Module.
Combines CRF model predictions with candidate rules, merges line items,
applies quality validation, and determines processing provenance.
"""

from typing import Dict, Any, List, Optional
import time
import re

from src.config import TARGET_FIELDS
from src.ml_extractor import InvoiceCRFExtractor
from src.candidate_extractor import CandidateExtractor, deduplicate_phrase, REJECT_PARTY_NAMES, INVALID_INV_NUMS
from src.line_item_extractor import LineItemExtractor
from src.validator import InvoiceValidator


class InvoiceProcessor:
    """Master orchestrator combining ML inference, candidate extraction, table extraction, and validation."""

    def __init__(self, crf_extractor: Optional[InvoiceCRFExtractor] = None):
        self.crf_extractor = crf_extractor or InvoiceCRFExtractor()
        self.candidate_extractor = CandidateExtractor()
        self.line_item_extractor = LineItemExtractor()
        self.validator = InvoiceValidator()

    def _sanitize_party(self, val: Any) -> Optional[str]:
        if not val:
            return None
        s = str(val).strip()
        s = re.sub(
            r"(?i)^(?:supplier\s+invoice\s+details|supplier|bill\s+to|ship\s+to|billed\s+to|customer|buyer)\s*[:#\-]?\s*",
            "",
            s,
        ).strip()
        s = deduplicate_phrase(s)
        if s.upper() in REJECT_PARTY_NAMES or len(s) < 2:
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
        Executes end-to-end extraction on document tokens and builds structured output.
        """
        start_time = time.time()
        filename = document_dict.get("filename", "unknown")
        doc_type = document_dict.get("doc_type", "unknown")
        page_count = document_dict.get("page_count", 1)

        # Build full text and line strings for candidate extractor
        full_text = " ".join(t["clean_text"] for t in tokens)
        
        # Build lines of text
        token_lines = self.line_item_extractor._group_tokens_by_y(tokens)
        text_lines = [" ".join(tok["clean_text"] for tok in line) for line in token_lines]

        # 1. Deterministic Candidate Extraction
        rule_candidates = self.candidate_extractor.extract_candidates_from_text(
            full_text, text_lines, tokens=tokens
        )

        # 2. Traditional ML CRF Extraction
        crf_fields = {}
        enriched_tokens = tokens
        extraction_method = "rule_based_fallback"
        model_trained = self.crf_extractor.is_trained()

        if model_trained and tokens:
            try:
                crf_fields, enriched_tokens = self.crf_extractor.extract_fields(tokens)
                if crf_fields:
                    extraction_method = "crf_ml"
            except Exception as e:
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
                if c_inv is not None:
                    final_fields[field] = c_inv
                elif r_inv is not None:
                    final_fields[field] = r_inv
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

        # Disclose fallback mode if model wasn't available
        if not model_trained:
            val_result["warnings"].insert(
                0, "CRF model is not trained or loaded. Extraction completed using rule-based candidate fallback."
            )

        duration_ms = round((time.time() - start_time) * 1000, 2)

        ocr_engine_used = "native_pdf" if doc_type == "digital_pdf" else "tesseract_ocr"

        result = {
            "metadata": {
                "filename": filename,
                "document_type": doc_type,
                "page_count": page_count,
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

        return result
