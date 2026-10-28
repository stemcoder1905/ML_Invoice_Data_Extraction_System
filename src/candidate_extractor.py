"""
Candidate Extractor Module (Deterministic Rules & Pattern Matching).
Serves as complementary extraction method to the CRF model, and provides
a reliable rule-based fallback when ML models are untrained or ambiguous.
"""

from typing import List, Dict, Any, Optional
import re
from src.config import PATTERNS, CURRENCY_MAP


class CandidateExtractor:
    """Extracts candidate invoice fields using anchored regular expressions and contextual lookarounds."""

    def __init__(self):
        self.re_gstin = re.compile(PATTERNS["gstin"])
        self.re_tax_id = re.compile(PATTERNS["tax_id"], re.IGNORECASE)
        self.re_inv = re.compile(PATTERNS["invoice_number"])
        self.re_po = re.compile(PATTERNS["po_number"])
        self.re_date = re.compile(PATTERNS["date"], re.IGNORECASE)
        self.re_currency = re.compile(PATTERNS["currency_code"])
        self.re_amount = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})|\d+(?:\.\d{1,2})?)")

    def extract_candidates_from_text(self, full_text: str, lines: List[str]) -> Dict[str, Any]:
        """Scans raw full document text and individual lines for invoice fields."""
        candidates: Dict[str, Any] = {}

        # 1. Invoice Number
        inv_match = self.re_inv.search(full_text)
        if inv_match:
            val = inv_match.group(1).strip()
            # Exclude false positives like "Date", "To"
            if len(val) >= 2 and not val.lower().startswith(("date", "page")):
                candidates["invoice_number"] = val

        # 2. Purchase Order
        po_match = self.re_po.search(full_text)
        if po_match:
            candidates["purchase_order_number"] = po_match.group(1).strip()

        # 3. Invoice Date
        for line in lines:
            line_clean = line.strip()
            if any(k in line_clean.lower() for k in ["invoice date", "inv date", "date of issue", "date:", "dated"]):
                date_match = self.re_date.search(line_clean)
                if date_match:
                    candidates["invoice_date"] = date_match.group(0).strip()
                    break
        if "invoice_date" not in candidates:
            # General date search in header lines (first 15 lines)
            for line in lines[:15]:
                date_match = self.re_date.search(line)
                if date_match:
                    candidates["invoice_date"] = date_match.group(0).strip()
                    break

        # 4. Tax ID / GSTIN
        gstin_match = self.re_gstin.search(full_text)
        if gstin_match:
            candidates["tax_id"] = gstin_match.group(0).strip()
        else:
            tax_match = self.re_tax_id.search(full_text)
            if tax_match:
                candidates["tax_id"] = tax_match.group(1).strip()

        # 5. Currency
        curr_code_match = self.re_currency.search(full_text)
        if curr_code_match:
            candidates["currency"] = curr_code_match.group(1).upper()
        else:
            for sym, code in CURRENCY_MAP.items():
                if sym in full_text:
                    candidates["currency"] = code
                    break

        # 6. Amounts: Subtotal, Tax Amount, Total Amount
        candidates.update(self._extract_amounts_from_lines(lines))

        # 7. Supplier and Buyer Names from anchors
        candidates.update(self._extract_parties_from_lines(lines))

        return candidates

    def _extract_amounts_from_lines(self, lines: List[str]) -> Dict[str, Any]:
        """Extracts financial totals by anchoring on keywords."""
        amounts: Dict[str, Any] = {}

        for line in lines:
            lower = line.lower()

            # Total amount
            if any(k in lower for k in ["grand total", "total amount", "net payable", "balance due", "total:"]) or (
                "total" in lower and "sub" not in lower and "tax" not in lower
            ):
                matches = self.re_amount.findall(line)
                if matches and "total_amount" not in amounts:
                    amounts["total_amount"] = matches[-1]

            # Subtotal
            elif any(k in lower for k in ["subtotal", "sub total", "sub-total", "taxable amount", "taxable value"]):
                matches = self.re_amount.findall(line)
                if matches and "subtotal" not in amounts:
                    amounts["subtotal"] = matches[-1]

            # Tax amount
            elif any(k in lower for k in ["tax amount", "total tax", "gst amount", "vat amount", "cgst", "sgst", "igst"]):
                matches = self.re_amount.findall(line)
                if matches and "tax_amount" not in amounts:
                    amounts["tax_amount"] = matches[-1]

        return amounts

    def _extract_parties_from_lines(self, lines: List[str]) -> Dict[str, Any]:
        """Extracts vendor (seller) and buyer from header sections."""
        parties: Dict[str, Any] = {}

        # Buyer search: looks for 'Bill To:', 'Billed To:', 'Customer:', 'Buyer:'
        for idx, line in enumerate(lines[:25]):
            lower = line.lower()
            if any(k in lower for k in ["bill to", "billed to", "sold to", "buyer", "customer", "ship to"]):
                # Clean prompt phrase
                clean_party = re.sub(r"(?i)(bill\s*to|billed\s*to|sold\s*to|buyer|customer|ship\s*to)\s*[:#\-]?\s*", "", line).strip()
                if clean_party and len(clean_party) > 2:
                    parties["buyer_name"] = clean_party
                    break
                elif idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    if next_line and not any(kw in next_line.lower() for kw in ["invoice", "date", "phone", "gst"]):
                        parties["buyer_name"] = next_line
                        break

        # Supplier search: typically in the top 5 lines if not explicitly labeled
        buyer = parties.get("buyer_name", "")
        for idx, line in enumerate(lines[:8]):
            line_str = line.strip()
            if not line_str:
                continue

            # If the line contains invoice number on the right (e.g. 'Vendor Name Invoice No: 123'), extract left segment
            candidate_line = line_str
            if re.search(r"(?i)\b(?:invoice|inv|bill)\s*(?:no|number|#)", line_str):
                left_part = re.split(r"(?i)\b(?:invoice|inv|bill)\s*(?:no|number|#)", line_str)[0].strip()
                if len(left_part) > 3 and not left_part.lower().startswith("tax"):
                    candidate_line = left_part

            lower = candidate_line.lower()
            if any(k in lower for k in ["tax invoice", "bill to", "billed to", "date:"]):
                continue

            # Don't pick buyer as supplier
            if buyer and (candidate_line.lower() == buyer.lower() or buyer.lower() in candidate_line.lower()):
                continue

            # If explicit label
            if any(k in lower for k in ["from:", "seller:", "vendor:"]):
                clean_sup = re.sub(r"(?i)(from|seller|vendor)\s*[:#\-]?\s*", "", candidate_line).strip()
                if clean_sup:
                    parties["supplier_name"] = clean_sup
                    break

            # Corporate name heuristics
            if any(sfx in lower for sfx in ["ltd", "limited", "pvt", "inc", "corp", "llc", "gmbh", "technologies", "services", "solutions", "enterprises", "supplies", "systems", "logistics"]):
                parties["supplier_name"] = candidate_line
                break

        # If supplier still not found, pick the first prominent non-empty line
        if "supplier_name" not in parties and lines:
            for line in lines[:5]:
                clean = line.strip()
                if clean and not any(k in clean.lower() for k in ["tax invoice", "date", "bill to"]):
                    if buyer and clean.lower() != buyer.lower():
                        parties["supplier_name"] = clean
                        break

        return parties
