"""
Machine Learning CRF Model Module for Invoice Information Extraction.
Trains, saves, loads, and executes sequence labeling using Conditional Random Fields.
Aggregates BIO entity chunks into structured invoice header fields.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import os
import joblib
import sklearn_crfsuite

from src.config import DEFAULT_MODEL_PATH
from src.feature_engineering import FeatureExtractor


# Mapping from BIO tag prefixes to invoice target fields
TAG_TO_FIELD = {
    "INV_NUM": "invoice_number",
    "INV_DATE": "invoice_date",
    "SUPPLIER": "supplier_name",
    "BUYER": "buyer_name",
    "PO_NUM": "purchase_order_number",
    "CURRENCY": "currency",
    "SUBTOTAL": "subtotal",
    "TAX_AMT": "tax_amount",
    "TOTAL": "total_amount",
    "TAX_ID": "tax_id",
}


class InvoiceCRFExtractor:
    """Traditional ML sequence labeler utilizing Conditional Random Fields."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.feature_extractor = FeatureExtractor()
        self.model: Optional[sklearn_crfsuite.CRF] = None
        self._load_if_exists()

    def _load_if_exists(self):
        """Attempts to load pre-trained CRF model from disk."""
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                self.model = None

    def is_trained(self) -> bool:
        """Checks if model is initialized and ready for inference."""
        return self.model is not None

    def train(
        self,
        X_train: List[List[Dict[str, Any]]],
        y_train: List[List[str]],
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Trains the CRF sequence labeling model on invoice document token sequences.
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=False,
        )
        self.model.fit(X_train, y_train)

        # Ensure model directory exists and save
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

        train_info = {
            "num_documents": len(X_train),
            "total_tokens": sum(len(seq) for seq in X_train),
            "classes": list(self.model.classes_),
            "model_path": str(self.model_path),
        }
        return train_info

    def predict_tokens(self, tokens: List[Dict[str, Any]]) -> List[str]:
        """Runs sequence prediction on token sequence."""
        if not self.is_trained():
            raise RuntimeError("CRF Model is not trained or loaded. Please train the model or use fallback mode.")

        features = self.feature_extractor.extract_sequence_features(tokens)
        preds = self.model.predict_single(features)
        return preds

    def extract_fields(self, tokens: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predicts labels and aggregates BIO tagged tokens into structured invoice fields.
        Returns extracted fields dictionary and enriched tokens with predicted labels.
        """
        if not tokens:
            return {}, []

        if not self.is_trained():
            # Return empty if model not trained (caller will employ fallback)
            return {}, tokens

        predictions = self.predict_tokens(tokens)
        enriched_tokens = []
        for t, pred in zip(tokens, predictions):
            tok_copy = dict(t)
            tok_copy["pred_label"] = pred
            enriched_tokens.append(tok_copy)

        # Chunk BIO sequences into entity values
        extracted_fields: Dict[str, Any] = {}
        current_entity_type = None
        current_tokens = []

        def commit_entity(ent_type: str, tok_list: List[Dict[str, Any]]):
            if not ent_type or not tok_list:
                return
            field_name = TAG_TO_FIELD.get(ent_type)
            if not field_name:
                return
            
            # Combine words into readable text
            text_val = " ".join(t["clean_text"] for t in tok_list).strip()
            # Clean common punctuation artifact at end
            text_val = text_val.rstrip(":,;")

            # For amounts or numbers, prefer the most complete single value
            if field_name not in extracted_fields or len(text_val) > len(str(extracted_fields[field_name])):
                extracted_fields[field_name] = text_val

        for t, label in zip(tokens, predictions):
            if label == "O":
                commit_entity(current_entity_type, current_tokens)
                current_entity_type = None
                current_tokens = []
            elif label.startswith("B-"):
                commit_entity(current_entity_type, current_tokens)
                current_entity_type = label[2:]
                current_tokens = [t]
            elif label.startswith("I-"):
                ent = label[2:]
                if ent == current_entity_type:
                    current_tokens.append(t)
                else:
                    commit_entity(current_entity_type, current_tokens)
                    current_entity_type = ent
                    current_tokens = [t]

        commit_entity(current_entity_type, current_tokens)
        return extracted_fields, enriched_tokens
