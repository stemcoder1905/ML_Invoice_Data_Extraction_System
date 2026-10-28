"""
Feature Engineering Module for CRF Sequence Labeling.
Extracts lexical, orthographic, domain-specific, spatial/positional,
and contextual window features from document tokens.
"""

from typing import List, Dict, Any
import re
from src.config import PATTERNS, CURRENCY_MAP


class FeatureExtractor:
    """Computes dense feature dictionaries for CRF sequence tagging."""

    def __init__(self):
        self.gstin_regex = re.compile(PATTERNS["gstin"])
        self.date_regex = re.compile(PATTERNS["date"])
        self.amount_regex = re.compile(PATTERNS["amount"])
        self.curr_regex = re.compile(PATTERNS["currency_code"])

        # Domain lexicons (case-insensitive)
        self.inv_keywords = {"invoice", "inv", "bill", "invoice#", "inv#", "factura", "bill#", "receipt"}
        self.date_keywords = {"date", "dated", "issue", "issued", "dt", "billing", "due"}
        self.po_keywords = {"po", "order", "purchase", "p.o.", "po#", "order#"}
        self.total_keywords = {"total", "amount", "payable", "grand", "due", "balance", "net"}
        self.subtotal_keywords = {"subtotal", "taxable", "sub-total", "taxable-value", "sub", "net-amount"}
        self.tax_keywords = {"tax", "vat", "gst", "cgst", "sgst", "igst", "hst", "pst", "cess"}
        self.vendor_keywords = {"from:", "seller", "vendor", "supplier", "merchant", "provider", "sold by"}
        self.buyer_keywords = {"to:", "bill to", "buyer", "customer", "client", "ship to", "billed to", "recipient"}
        self.company_suffixes = {"ltd", "limited", "pvt", "inc", "corp", "corporation", "llc", "co", "enterprises", "solutions", "technologies", "services", "gmbh"}

    @staticmethod
    def word_shape(text: str) -> str:
        """
        Generates simplified word shape (e.g. 'INV-2024-01' -> 'XXX-9999-99').
        Collapses consecutive repeats: 'XXXX' -> 'X+'.
        """
        shape = []
        for ch in text:
            if ch.isupper():
                shape.append("X")
            elif ch.islower():
                shape.append("x")
            elif ch.isdigit():
                shape.append("9")
            else:
                shape.append(ch)
        raw_shape = "".join(shape)
        # Collapse runs of 4+ identical characters
        return re.sub(r"X{4,}", "X+", re.sub(r"x{4,}", "x+", re.sub(r"9{4,}", "9+", raw_shape)))

    def token_to_features(self, tokens: List[Dict[str, Any]], i: int) -> Dict[str, Any]:
        """Computes comprehensive feature vector for token at index i in document token sequence."""
        t = tokens[i]
        word = t["clean_text"]
        word_lower = word.lower()
        norm_bbox = t.get("norm_bbox", [0.0, 0.0, 0.0, 0.0])
        x0, y0, x1, y1 = norm_bbox
        w = max(0.0, x1 - x0)
        h = max(0.0, y1 - y0)

        # Base lexical & orthographic features
        features = {
            "bias": 1.0,
            "word.lower": word_lower,
            "word.len": len(word),
            "word.isupper": word.isupper(),
            "word.islower": word.islower(),
            "word.istitle": word.istitle(),
            "word.isdigit": word.isdigit(),
            "word.isalnum": word.isalnum(),
            "word.has_digit": any(c.isdigit() for c in word),
            "word.has_hyphen": "-" in word,
            "word.has_slash": "/" in word,
            "word.has_colon": ":" in word,
            "word.shape": self.word_shape(word),
            "prefix_2": word_lower[:2] if len(word) >= 2 else word_lower,
            "prefix_3": word_lower[:3] if len(word) >= 3 else word_lower,
            "suffix_2": word_lower[-2:] if len(word) >= 2 else word_lower,
            "suffix_3": word_lower[-3:] if len(word) >= 3 else word_lower,

            # Domain keyword matches
            "kw.invoice": word_lower in self.inv_keywords,
            "kw.date": word_lower in self.date_keywords,
            "kw.po": word_lower in self.po_keywords,
            "kw.total": word_lower in self.total_keywords,
            "kw.subtotal": word_lower in self.subtotal_keywords,
            "kw.tax": word_lower in self.tax_keywords,
            "kw.company_sfx": word_lower in self.company_suffixes,
            "kw.currency": (word in CURRENCY_MAP) or bool(self.curr_regex.match(word)),

            # Pattern matches
            "pat.gstin": bool(self.gstin_regex.match(word)),
            "pat.date": bool(self.date_regex.match(word)),
            "pat.amount": bool(self.amount_regex.match(word)),

            # Spatial / Positional features
            "pos.page_num": t.get("page_num", 1),
            "pos.is_page_1": t.get("page_num", 1) == 1,
            "pos.x0_bin": round(x0 * 20) / 20.0,  # 5% grid bins
            "pos.y0_bin": round(y0 * 20) / 20.0,
            "pos.w_bin": round(w * 20) / 20.0,
            "pos.h_bin": round(h * 20) / 20.0,
            "pos.is_top_quarter": y0 < 0.25,
            "pos.is_top_half": y0 < 0.50,
            "pos.is_bottom_quarter": y0 > 0.75,
            "pos.is_left_half": x0 < 0.50,
            "pos.is_right_half": x0 >= 0.50,
        }

        # Contextual window features: Previous token (i - 1)
        if i > 0:
            prev_t = tokens[i - 1]
            p_word = prev_t["clean_text"]
            p_lower = p_word.lower()
            p_x0, p_y0, p_x1, p_y1 = prev_t.get("norm_bbox", [0.0, 0.0, 0.0, 0.0])

            features.update({
                "-1:word.lower": p_lower,
                "-1:word.istitle": p_word.istitle(),
                "-1:word.isupper": p_word.isupper(),
                "-1:word.isdigit": p_word.isdigit(),
                "-1:word.has_colon": ":" in p_word,
                "-1:word.shape": self.word_shape(p_word),
                "-1:kw.invoice": p_lower in self.inv_keywords,
                "-1:kw.date": p_lower in self.date_keywords,
                "-1:kw.po": p_lower in self.po_keywords,
                "-1:kw.total": p_lower in self.total_keywords,
                "-1:kw.subtotal": p_lower in self.subtotal_keywords,
                "-1:kw.tax": p_lower in self.tax_keywords,
                "-1:kw.vendor": any(k in p_lower for k in self.vendor_keywords),
                "-1:kw.buyer": any(k in p_lower for k in self.buyer_keywords),
                "-1:same_line": abs(y0 - p_y0) < 0.015,
                "-1:dx": round((x0 - p_x1) * 20) / 20.0,
            })
        else:
            features["BOS"] = True  # Beginning of sequence

        # Contextual window features: Previous-previous token (i - 2)
        if i > 1:
            p2_t = tokens[i - 2]
            p2_lower = p2_t["clean_text"].lower()
            features.update({
                "-2:word.lower": p2_lower,
                "-2:kw.invoice": p2_lower in self.inv_keywords,
                "-2:kw.date": p2_lower in self.date_keywords,
                "-2:kw.total": p2_lower in self.total_keywords,
                "-2:kw.tax": p2_lower in self.tax_keywords,
            })

        # Contextual window features: Next token (i + 1)
        if i < len(tokens) - 1:
            next_t = tokens[i + 1]
            n_word = next_t["clean_text"]
            n_lower = n_word.lower()
            n_x0, n_y0, n_x1, n_y1 = next_t.get("norm_bbox", [0.0, 0.0, 0.0, 0.0])

            features.update({
                "+1:word.lower": n_lower,
                "+1:word.istitle": n_word.istitle(),
                "+1:word.isupper": n_word.isupper(),
                "+1:word.isdigit": n_word.isdigit(),
                "+1:word.shape": self.word_shape(n_word),
                "+1:kw.company_sfx": n_lower in self.company_suffixes,
                "+1:same_line": abs(y0 - n_y0) < 0.015,
                "+1:dx": round((n_x0 - x1) * 20) / 20.0,
            })
        else:
            features["EOS"] = True  # End of sequence

        # Contextual window features: Next-next token (i + 2)
        if i < len(tokens) - 2:
            n2_lower = tokens[i + 2]["clean_text"].lower()
            features.update({
                "+2:word.lower": n2_lower,
                "+2:kw.company_sfx": n2_lower in self.company_suffixes,
            })

        return features

    def extract_sequence_features(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts feature dictionaries for an entire sequence of document tokens."""
        return [self.token_to_features(tokens, i) for i in range(len(tokens))]
