"""
Candidate Extractor Module (Deterministic Rules & Pattern Matching).
Serves as complementary extraction method to the CRF model, and provides
a reliable rule-based fallback when ML models are untrained or ambiguous.
"""

from typing import List, Dict, Any, Optional
import re
from src.config import PATTERNS, CURRENCY_MAP

REJECT_PARTY_NAMES = {
    "SUPPLIER", "BILL TO", "BILLED TO", "SOLD TO", "BUYER", "CUSTOMER",
    "SHIP TO", "SHIPPED TO", "INVOICE", "TAX INVOICE", "DATE", "TAX ID",
    "PURCHASE ORDER", "ORDER", "CURRENCY", "DESCRIPTION", "TOTAL",
}


class CandidateExtractor:
    """Extracts candidate invoice fields using anchored regular expressions, spatial layout, and lookarounds."""

    def __init__(self):
        self.re_gstin = re.compile(PATTERNS["gstin"])
        self.re_tax_id = re.compile(PATTERNS["tax_id"], re.IGNORECASE)
        self.re_inv = re.compile(PATTERNS["invoice_number"])
        self.re_po = re.compile(PATTERNS["po_number"])
        self.re_date = re.compile(PATTERNS["date"], re.IGNORECASE)
        self.re_currency = re.compile(PATTERNS["currency_code"])
        self.re_amount = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})|\d+(?:\.\d{1,2})?)")

    def extract_candidates_from_text(
        self,
        full_text: str,
        lines: List[str],
        tokens: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Scans raw full document text, individual lines, and spatial tokens for invoice fields."""
        candidates: Dict[str, Any] = {}

        # 1. Invoice Number
        inv_match = self.re_inv.search(full_text)
        if inv_match:
            val = inv_match.group(1).strip()
            # Exclude false positives like "Date", "To", "Number"
            if len(val) >= 2 and not val.lower().startswith(("date", "page", "number")):
                candidates["invoice_number"] = val
        if "invoice_number" not in candidates:
            for line in lines:
                m = self.re_inv.search(line)
                if m:
                    val = m.group(1).strip()
                    if len(val) >= 2 and not val.lower().startswith(("date", "page", "number")):
                        candidates["invoice_number"] = val
                        break

        # 2. Purchase Order
        po_match = self.re_po.search(full_text)
        if po_match:
            candidates["purchase_order_number"] = po_match.group(1).strip()
        else:
            for line in lines:
                m = self.re_po.search(line)
                if m:
                    candidates["purchase_order_number"] = m.group(1).strip()
                    break

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
                cand_id = tax_match.group(1).strip()
                if not cand_id.startswith(("PO-", "PO ", "SO-")):
                    candidates["tax_id"] = cand_id

        if "tax_id" not in candidates or candidates.get("tax_id", "").startswith(("PO-", "PO ")):
            for line in lines:
                m = re.search(r"(?i)\b(?:tax\s*id|ein|vat\s*id|tax\s*no)\s*[:#\-]?\s*([A-Z0-9\-]{7,18})\b", line)
                if m:
                    cand = m.group(1).strip()
                    if not cand.startswith(("PO-", "PO ")):
                        candidates["tax_id"] = cand
                        break

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

        # 7. Supplier and Buyer Names (spatial token awareness + anchor heuristics)
        parties = {}
        if tokens:
            parties = self._extract_parties_from_tokens(tokens)
        if not parties.get("supplier_name") or not parties.get("buyer_name"):
            line_parties = self._extract_parties_from_lines(lines)
            for k, v in line_parties.items():
                if k not in parties or not parties[k]:
                    parties[k] = v

        candidates.update(parties)
        return candidates

    def _extract_amounts_from_lines(self, lines: List[str]) -> Dict[str, Any]:
        """Extracts financial totals by anchoring on keywords across lines."""
        amounts: Dict[str, Any] = {}

        def get_amount(curr_idx: int) -> Optional[str]:
            line_str = lines[curr_idx]
            # Strip percentage values like (8%) so they are not mistaken for amounts
            clean_l = re.sub(r"\(\s*\d+(?:\.\d+)?%\s*\)", "", line_str)
            m = self.re_amount.findall(clean_l)
            if m:
                return m[-1]
            if curr_idx + 1 < len(lines):
                next_l = re.sub(r"\(\s*\d+(?:\.\d+)?%\s*\)", "", lines[curr_idx + 1])
                next_m = self.re_amount.findall(next_l)
                if next_m:
                    return next_m[-1]
            return None

        for idx, line in enumerate(lines):
            lower = line.lower()

            # Total amount
            if any(k in lower for k in ["grand total", "total amount", "net payable", "balance due", "total due", "amount due", "total:"]) or (
                "total" in lower and "sub" not in lower and "tax" not in lower and "line" not in lower
            ):
                amt = get_amount(idx)
                if amt and "total_amount" not in amounts:
                    amounts["total_amount"] = amt

            # Subtotal
            elif any(k in lower for k in ["subtotal", "sub total", "sub-total", "taxable amount", "taxable value", "net amount"]):
                amt = get_amount(idx)
                if amt and "subtotal" not in amounts:
                    amounts["subtotal"] = amt

            # Tax amount
            elif any(k in lower for k in ["tax amount", "total tax", "gst amount", "vat amount", "cgst", "sgst", "igst"]) or (
                re.search(r"\b(?:tax|vat|gst)\b(?:\s*\(\s*\d+%\s*\))?", lower)
                and "sub" not in lower
                and "id" not in lower
                and "no" not in lower
                and "number" not in lower
                and "gstin" not in lower
            ):
                amt = get_amount(idx)
                if amt and "tax_amount" not in amounts:
                    amounts["tax_amount"] = amt

        # Mathematical derivation if 2 of 3 amounts are present
        if "total_amount" in amounts and "subtotal" in amounts and "tax_amount" not in amounts:
            try:
                tot = float(amounts["total_amount"].replace(",", ""))
                sub = float(amounts["subtotal"].replace(",", ""))
                if tot >= sub:
                    amounts["tax_amount"] = f"{round(tot - sub, 2):.2f}"
            except Exception:
                pass
        elif "total_amount" in amounts and "tax_amount" in amounts and "subtotal" not in amounts:
            try:
                tot = float(amounts["total_amount"].replace(",", ""))
                tax = float(amounts["tax_amount"].replace(",", ""))
                if tot >= tax:
                    amounts["subtotal"] = f"{round(tot - tax, 2):.2f}"
            except Exception:
                pass
        elif "subtotal" in amounts and "tax_amount" in amounts and "total_amount" not in amounts:
            try:
                sub = float(amounts["subtotal"].replace(",", ""))
                tax = float(amounts["tax_amount"].replace(",", ""))
                amounts["total_amount"] = f"{round(sub + tax, 2):.2f}"
            except Exception:
                pass

        return amounts

    def _extract_parties_from_tokens(self, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Spatially extracts supplier and buyer from 2-column or block-structured headers."""
        parties: Dict[str, Any] = {}
        if not tokens:
            return parties

        p1_tokens = [t for t in tokens if t.get("page_num", 1) == 1]
        sup_headers = [
            t for t in p1_tokens
            if t["clean_text"].upper() in ["SUPPLIER", "FROM:", "SELLER:", "VENDOR:"] and t["norm_bbox"][1] < 0.35
        ]
        buyer_headers = [
            t for t in p1_tokens
            if t["clean_text"].upper() in ["BILL", "BILLED", "BUYER", "CUSTOMER:", "SHIP"] and t["norm_bbox"][1] < 0.35
        ]

        if sup_headers and buyer_headers:
            sup_t = sup_headers[0]
            buyer_t = buyer_headers[0]
            col_split_x = buyer_t["norm_bbox"][0] - 0.04

            # Supplier candidates: below SUPPLIER, left of column divider
            sup_cands = [
                t for t in p1_tokens
                if t["norm_bbox"][0] < col_split_x
                and t["norm_bbox"][1] > sup_t["norm_bbox"][3]
                and t["norm_bbox"][1] < 0.35
                and t["clean_text"].upper() not in ["SUPPLIER", "TAX", "ID:", "PURCHASE", "ORDER:", "PO:", "BILL", "TO"]
                and not re.match(r"^(?:tax|id|po|order|tel|phone|gstin)", t["clean_text"], re.I)
            ]

            # Buyer candidates: below BILL TO, aligned with buyer header (x >= buyer_t.x0 - 0.04)
            buyer_cands = [
                t for t in p1_tokens
                if t["norm_bbox"][0] >= (buyer_t["norm_bbox"][0] - 0.04)
                and t["norm_bbox"][1] > buyer_t["norm_bbox"][3]
                and t["norm_bbox"][1] < 0.35
                and t["clean_text"].upper() not in ["BILL", "TO", "CUSTOMER", "BUYER", "CURRENCY:", "USD", "EUR", "INR", "PO:"]
                and not re.match(r"^(?:currency|po|order|tel|phone|attn)", t["clean_text"], re.I)
            ]

            if sup_cands:
                first_y = sup_cands[0]["norm_bbox"][1]
                first_line_sup = [t for t in sup_cands if abs(t["norm_bbox"][1] - first_y) < 0.007]
                first_line_sup.sort(key=lambda x: x["norm_bbox"][0])
                cand_sup = " ".join(t["clean_text"] for t in first_line_sup).strip()
                if cand_sup.upper() not in REJECT_PARTY_NAMES and len(cand_sup) > 2:
                    parties["supplier_name"] = cand_sup

            if buyer_cands:
                first_y = buyer_cands[0]["norm_bbox"][1]
                first_line_buyer = [t for t in buyer_cands if abs(t["norm_bbox"][1] - first_y) < 0.007]
                first_line_buyer.sort(key=lambda x: x["norm_bbox"][0])
                cand_buyer = " ".join(t["clean_text"] for t in first_line_buyer).strip()
                if cand_buyer.upper() not in REJECT_PARTY_NAMES and len(cand_buyer) > 2:
                    parties["buyer_name"] = cand_buyer

        return parties

    def _extract_parties_from_lines(self, lines: List[str]) -> Dict[str, Any]:
        """Extracts vendor (seller) and buyer from header sections."""
        parties: Dict[str, Any] = {}

        # Buyer search: looks for 'Bill To:', 'Billed To:', 'Customer:', 'Buyer:'
        for idx, line in enumerate(lines[:25]):
            lower = line.lower()
            if any(k in lower for k in ["bill to", "billed to", "sold to", "buyer", "customer", "ship to"]):
                # Clean prompt phrase
                clean_party = re.sub(r"(?i)(bill\s*to|billed\s*to|sold\s*to|buyer|customer|ship\s*to)\s*[:#\-]?\s*", "", line).strip()
                if clean_party.upper() not in REJECT_PARTY_NAMES and len(clean_party) > 2:
                    parties["buyer_name"] = clean_party
                    break
                elif idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    if next_line.upper() not in REJECT_PARTY_NAMES and len(next_line) > 2 and not any(kw in next_line.lower() for kw in ["invoice", "date", "phone", "gst"]):
                        parties["buyer_name"] = next_line
                        break

        # Supplier search: typically in the top 8 lines
        buyer = parties.get("buyer_name", "")
        for idx, line in enumerate(lines[:12]):
            line_str = line.strip()
            if not line_str or line_str.upper() in REJECT_PARTY_NAMES:
                continue

            # If the line contains invoice number on the right, extract left segment
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
            if any(k in lower for k in ["from:", "seller:", "vendor:", "supplier:"]):
                clean_sup = re.sub(r"(?i)(from|seller|vendor|supplier)\s*[:#\-]?\s*", "", candidate_line).strip()
                if clean_sup.upper() not in REJECT_PARTY_NAMES and len(clean_sup) > 2:
                    parties["supplier_name"] = clean_sup
                    break

            # Corporate name heuristics
            if any(sfx in lower for sfx in ["ltd", "limited", "pvt", "inc", "corp", "llc", "gmbh", "s.l.", "pte", "technologies", "services", "solutions", "enterprises", "supplies", "systems", "logistics"]):
                if candidate_line.upper() not in REJECT_PARTY_NAMES:
                    parties["supplier_name"] = candidate_line
                    break

        # Fallback for supplier: first prominent non-header line
        if "supplier_name" not in parties and lines:
            for line in lines[:6]:
                clean = line.strip()
                if clean and clean.upper() not in REJECT_PARTY_NAMES and not any(k in clean.lower() for k in ["tax invoice", "date", "bill to", "invoice"]):
                    if buyer and clean.lower() != buyer.lower():
                        parties["supplier_name"] = clean
                        break

        return parties
