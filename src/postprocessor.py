"""
Postprocessing and Ensemble Merging Module.
Combines CRF model predictions with candidate rules, merges line items,
applies quality validation, and determines processing provenance.
"""

from typing import Dict, Any, List, Optional
import time

from src.config import TARGET_FIELDS
from src.ml_extractor import InvoiceCRFExtractor
from src.candidate_extractor import CandidateExtractor
from src.line_item_extractor import LineItemExtractor
from src.validator import InvoiceValidator


class InvoiceProcessor:
    """Master orchestrator combining ML inference, candidate extraction, table extraction, and validation."""

    def __init__(self, crf_extractor: Optional[InvoiceCRFExtractor] = None):
        self.crf_extractor = crf_extractor or InvoiceCRFExtractor()
        self.candidate_extractor = CandidateExtractor()
        self.line_item_extractor = LineItemExtractor()
        self.validator = InvoiceValidator()

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

        # 3. Hybrid Merge (Complementary rules for fields CRF missed or unconfident on)
        final_fields: Dict[str, Any] = {}
        for field in TARGET_FIELDS:
            crf_val = crf_fields.get(field)
            rule_val = rule_candidates.get(field)

            if crf_val is not None and str(crf_val).strip() != "":
                final_fields[field] = crf_val
            elif rule_val is not None:
                final_fields[field] = rule_val
                if model_trained and extraction_method == "crf_ml":
                    extraction_method = "hybrid_crf_rules"
            else:
                final_fields[field] = None

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
