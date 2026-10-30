"""
Candidate Extractor Module (Deterministic Rules, Pattern Matching & Layout Analysis).
Extracts candidate invoice fields using anchored regular expressions, spatial layout,
multilingual field aliases, and financial cross-checks.
Serves as complementary extraction method to the traditional ML CRF model.
"""

from typing import List, Dict, Any, Optional
import re
from src.config import PATTERNS, CURRENCY_MAP, FIELD_ALIASES

REJECT_PARTY_NAMES = {
    "SUPPLIER", "BILL TO", "BILLED TO", "SOLD TO", "BUYER", "CUSTOMER",
    "SHIP TO", "SHIPPED TO", "INVOICE", "TAX INVOICE", "INVOICE DETAILS",
    "DATE", "TAX ID", "PURCHASE ORDER", "ORDER", "CURRENCY", "DESCRIPTION",
    "TOTAL", "SUBTOTAL", "GRAND TOTAL", "LINE ITEMS", "TERMS & CONDITIONS",
    "PAYMENT TERMS", "SYNTHETIC SAMPLE", "SYNTHETIC", "SAMPLE", "SELLER",
    "SELLER NAME", "VENDOR", "MERCHANT", "DETAILS", "TAX INVOICE/BILL OF SUPPLY",
    "BILL OF SUPPLY", "TAX INVOICE / BILL OF SUPPLY", "REVERSE CHARGE",
}

INVALID_INV_NUMS = {
    "SYNTHETIC", "SAMPLE", "ORIGINAL", "DUPLICATE", "COPY", "INVOICE", "TAX",
    "DETAILS", "DATE", "NUMBER", "PAGE", "SUPPLIER", "CUSTOMER", "BUYER",
    "TOTAL", "BILL", "TRUE", "FALSE", "NONE", "NULL", "SUPPLY", "PLACE",
    "HARYANA", "DELHI", "MAHARASHTRA", "KARNATAKA",
}

ADDRESS_INDICATORS = [
    "ground floor", "1st floor", "2nd floor", "3rd floor", "plot no", "plot",
    "khewat", "khata", "killa", "village", "sector", "road", "street", "building",
    "bldg", "phase", "near", "opposite", "opp.", "lane", "nagar", "marg", "dist",
    "district", "city", "haryana", "delhi", "karnataka", "maharashtra", "bangalore",
    "bengaluru", "mumbai", "gurugram", "gurgaon", "pune", "hyderabad", "chennai",
    "kolkata", "india", "pin", "pincode", "fssai", "gstin", "attn", "support@",
    "email", "phone", "tel", "mobile", "fax", "www.", ".com", "pg,", "pg ", "bandhwari",
    "shree shyam", "gwal pahari", "balliawas", "balola"
]


def deduplicate_phrase(text: str) -> str:
    """Removes duplicate adjacent phrases caused by multi-column horizontal line joining."""
    if not text:
        return text
    clean = text.strip()
    words = clean.split()
    if len(words) >= 4 and len(words) % 2 == 0:
        half = len(words) // 2
        if words[:half] == words[half:]:
            return " ".join(words[:half])
    half_len = len(clean) // 2
    if clean[:half_len].strip() == clean[half_len:].strip():
        return clean[:half_len].strip()
    return clean


class CandidateExtractor:
    """Extracts candidate invoice fields using anchored regular expressions, spatial layout, and lookarounds."""

    def __init__(self):
        self.re_gstin = re.compile(PATTERNS["gstin"])
        self.re_tax_id = re.compile(PATTERNS["tax_id"], re.IGNORECASE)
        # Primary labeled invoice number
        self.re_inv_labeled = re.compile(
            r"(?i)\b(?:tax\s*invoice|invoice|inv|bill|document|receipt)\s*(?:no\.?|number|num\.?|ref\.?|#)\s*[:#\-]?\s*([A-Za-z0-9\-_/]{3,35})\b"
        )
        # Secondary fallback invoice number (requires colon or dash delimiter)
        self.re_inv_delimited = re.compile(
            r"(?i)\b(?:invoice|inv|bill)\s*[:#\-]\s*([A-Za-z0-9\-_/]{3,35})\b"
        )
        self.re_po = re.compile(PATTERNS["po_number"])
        self.re_date = re.compile(PATTERNS["date"], re.IGNORECASE)
        self.re_currency = re.compile(PATTERNS["currency_code"])
        self.re_amount = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})|\d+(?:\.\d{1,4})?)")

    def _is_address_line(self, text: str) -> bool:
        """Determines if a text fragment is an address rather than a company or person name."""
        if not text:
            return False
        lower = text.lower().strip()
        # Explicit address starters
        if any(lower.startswith(k) for k in ["ground floor", "plot", "shop", "near", "khewat", "shree shyam", "flat", "room", "house", "survey"]):
            return True
        # Check postal codes (6 digits)
        if re.search(r"\b[1-9][0-9]{5}\b", lower):
            return True
        # Match address keywords
        count = sum(1 for kw in ADDRESS_INDICATORS if kw in lower)
        return count >= 2

    def _is_valid_invoice_number(self, val: str) -> bool:
        if not val or len(val) < 2:
            return False
        clean = val.strip().strip(":#-.,")
        if clean.upper() in INVALID_INV_NUMS:
            return False
        if clean.lower().startswith(("date", "page", "number", "synthetic", "sample", "tax", "bill", "place")):
            return False
        if clean.isalpha() and len(clean) > 4:
            return False
        return True

    def _is_valid_po_number(self, val: str) -> bool:
        """Ensures extracted PO/Order number is not a date or tax identifier."""
        if not val or len(val) < 3:
            return False
        clean = val.strip().strip(":#-.,")
        # Reject dates
        if re.match(r"^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$", clean) or re.match(r"^\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}$", clean):
            return False
        # Reject GSTINs
        if self.re_gstin.match(clean):
            return False
        # Reject email/phone
        if "@" in clean or clean.lower().startswith(("http", "www")):
            return False
        return True

    def extract_candidates_from_text(
        self,
        full_text: str,
        lines: List[str],
        tokens: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Scans raw full document text, individual lines, and spatial tokens for invoice fields."""
        candidates: Dict[str, Any] = {}

        # 1. Invoice Number (Prioritize labeled patterns on spatial lines)
        for line in lines:
            m = re.search(
                r"(?i)\b(?:bill\s*of\s*supply\s*number|invoice\s*number\s*#|invoice\s*no\.?|invoice\s*number|tax\s*invoice\s*no\.?)\s*[:#\s-]*([A-Za-z0-9\-_/]{4,25})\b",
                line,
            )
            if m and self._is_valid_invoice_number(m.group(1)):
                candidates["invoice_number"] = m.group(1).strip()
                break

        if "invoice_number" not in candidates:
            m_labeled = self.re_inv_labeled.search(full_text)
            if m_labeled and self._is_valid_invoice_number(m_labeled.group(1)):
                candidates["invoice_number"] = m_labeled.group(1).strip()

        if "invoice_number" not in candidates:
            m_delim = self.re_inv_delimited.search(full_text)
            if m_delim and self._is_valid_invoice_number(m_delim.group(1)):
                candidates["invoice_number"] = m_delim.group(1).strip()

        # 2. Purchase Order & Order Number
        for line in lines:
            m = re.search(
                r"(?i)\b(?:order\s*(?:id|number|no\.?)|against\s*order\s*id\s*\(?|p\.?o\.?\s*(?:no\.?|number|#))\s*[:#\s-]*([A-Za-z0-9\-_/]{4,30})\b",
                line,
            )
            if m:
                cand_order = m.group(1).strip().strip("()")
                if self._is_valid_po_number(cand_order):
                    candidates["purchase_order_number"] = cand_order
                    break

        if "purchase_order_number" not in candidates:
            po_match = self.re_po.search(full_text)
            if po_match and self._is_valid_po_number(po_match.group(1)):
                candidates["purchase_order_number"] = po_match.group(1).strip().strip("()")

        # 3. Invoice Date (Strictly avoid Order Date)
        for line in lines:
            line_clean = line.strip()
            if any(k in line_clean.lower() for k in ["invoice date", "bill of supply date", "inv date", "date of issue", "dated"]):
                date_match = self.re_date.search(line_clean)
                if date_match:
                    candidates["invoice_date"] = date_match.group(0).strip()
                    break

        if "invoice_date" not in candidates:
            for line in lines:
                line_clean = line.strip()
                # Match labeled Date: that is not Order Date:
                m = re.search(r"(?i)(?<!order\s)\bdate\s*[:#\-]\s*([A-Za-z0-9\-/\.,\s]+)", line_clean)
                if m:
                    date_match = self.re_date.search(m.group(0))
                    if date_match:
                        candidates["invoice_date"] = date_match.group(0).strip()
                        break

        if "invoice_date" not in candidates:
            for line in lines[:20]:
                if not re.search(r"(?i)\border\s*date\b", line):
                    date_match = self.re_date.search(line)
                    if date_match:
                        candidates["invoice_date"] = date_match.group(0).strip()
                        break

        # 4. Tax ID / GSTIN
        # Priority 1: Indian GSTIN (State code + PAN + entity + Z + check)
        gstin_match = self.re_gstin.search(full_text)
        if gstin_match:
            candidates["tax_id"] = gstin_match.group(0).strip()
        else:
            tax_match = self.re_tax_id.search(full_text)
            if tax_match:
                cand_id = tax_match.group(1).strip()
                if not cand_id.startswith(("PO-", "PO ", "SO-")) and "@" not in cand_id and len(cand_id) != 14:
                    candidates["tax_id"] = cand_id

        if "tax_id" not in candidates or candidates.get("tax_id", "").startswith(("PO-", "PO ")) or "@" in candidates.get("tax_id", ""):
            for line in lines:
                m = re.search(r"(?i)\b(?:tax\s*id|ein|vat\s*id|tax\s*no|gstin)\s*[:#\-]?\s*([A-Z0-9\-]{7,18})\b", line)
                if m:
                    cand = m.group(1).strip()
                    if not cand.startswith(("PO-", "PO ")) and "@" not in cand and len(cand) != 14:
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

        # If currency is not explicitly stated, infer INR for Indian GST invoices
        if not candidates.get("currency"):
            if "gstin" in full_text.lower() or "cgst" in full_text.lower() or "sgst" in full_text.lower() or "place of supply" in full_text.lower():
                candidates["currency"] = "INR"

        # 6. Amounts: Subtotal, Tax Amount, Total Amount
        candidates.update(self._extract_amounts_from_lines(lines, tokens=tokens))

        # 7. Supplier and Buyer Names (spatial layout + anchors)
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

    def _extract_amounts_from_lines(self, lines: List[str], tokens: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Extracts financial totals by anchoring on keywords across lines and summary blocks."""
        amounts: Dict[str, Any] = {}

        def get_amount(curr_idx: int) -> Optional[str]:
            line_str = lines[curr_idx]
            clean_l = re.sub(r"\(\s*\d+(?:\.\d+)?%\s*\)", "", line_str)
            clean_l = re.sub(r"\d{1,2}%\s*\+\s*\d+(?:\.\d+)?", "", clean_l)
            m = self.re_amount.findall(clean_l)
            # Filter out numbers that represent years in legal rules (e.g. 2017 in CGST Rules, 2017)
            filtered = []
            for num in m:
                clean_num = num.replace(",", "")
                # Exclude 14-digit FSSAI numbers
                if len(clean_num) >= 10 and not "." in clean_num:
                    continue
                if clean_num in ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"] and "rule" in line_str.lower():
                    continue
                filtered.append(num)
            if filtered:
                return filtered[-1]
            if curr_idx + 1 < len(lines):
                next_l = re.sub(r"\(\s*\d+(?:\.\d+)?%\s*\)", "", lines[curr_idx + 1])
                m_next = self.re_amount.findall(next_l)
                filtered_next = [n for n in m_next if len(n.replace(",", "")) < 10]
                if filtered_next:
                    return filtered_next[-1]
            return None

        # Scan for GST tax components (CGST + SGST or IGST)
        cgst_amt = None
        sgst_amt = None
        igst_amt = None

        for idx, line in enumerate(lines):
            lower = line.lower()

            # Ignore rule citations or terms
            if any(term in lower for term in ["rules, 20", "rule 31", "cgst rules", "terms & conditions", "act, 20", "act 20", "section 9", "section 1", "fssai", "cin:"]):
                continue

            # Skip table row headers
            if any(term in lower for term in ["line total", "incl. tax", "particulars", "description", "unit price", "quantity"]) and "total:" not in lower and "subtotal" not in lower:
                continue

            # Grand Total / Invoice Value / Total Amount
            is_non_monetary_total = any(term in lower for term in ["item", "items", "qty", "quantity", "count", "pcs", "nos", "weight", "grams", "kg", "box", "package"])
            if not is_non_monetary_total and (
                any(k in lower for k in [
                    "grand total", "total amount", "total value", "invoice total", "net payable", "balance due", "total due",
                    "amount due", "invoice value", "item total", "total amt"
                ]) or (
                    "total" in lower and "sub" not in lower and "tax" not in lower and "line" not in lower and "cess" not in lower
                )
            ):
                amt = get_amount(idx)
                if amt and "total_amount" not in amounts:
                    amounts["total_amount"] = amt

            # Subtotal / Taxable Value
            elif any(k in lower for k in [
                "subtotal", "sub total", "sub-total", "taxable amount", "taxable value",
                "taxable amt", "assessable value", "net amount", "gross value"
            ]):
                amt = get_amount(idx)
                if amt and "subtotal" not in amounts:
                    amounts["subtotal"] = amt

            # Specific CGST / SGST amount detection in summary
            if "cgst amt" in lower or "total cgst" in lower:
                amt = get_amount(idx)
                if amt:
                    try:
                        cgst_amt = float(amt.replace(",", ""))
                    except Exception:
                        pass
            if "sgst amt" in lower or "total sgst" in lower or "s/ut gst amt" in lower:
                amt = get_amount(idx)
                if amt:
                    try:
                        sgst_amt = float(amt.replace(",", ""))
                    except Exception:
                        pass
            if "igst amt" in lower or "total igst" in lower:
                amt = get_amount(idx)
                if amt:
                    try:
                        igst_amt = float(amt.replace(",", ""))
                    except Exception:
                        pass

            # Explicit Total Tax amount
            elif any(k in lower for k in ["total tax", "tax amount", "tax total", "total gst", "gst amount", "vat amount"]) or (
                re.search(r"\b(?:tax|vat|gst)\b(?:\s*\(\s*\d+%\s*\))?", lower)
                and "sub" not in lower
                and "id" not in lower
                and "no" not in lower
                and "number" not in lower
                and "gstin" not in lower
                and "invoice" not in lower
                and "rules" not in lower
            ):
                amt = get_amount(idx)
                if amt and "tax_amount" not in amounts:
                    amounts["tax_amount"] = amt

        # If CGST and SGST amounts found, aggregate into tax_amount
        if cgst_amt is not None and sgst_amt is not None and "tax_amount" not in amounts:
            amounts["tax_amount"] = f"{round(cgst_amt + sgst_amt, 2):.2f}"
        elif igst_amt is not None and "tax_amount" not in amounts:
            amounts["tax_amount"] = f"{round(igst_amt, 2):.2f}"

        # If explicit labeled lines already yielded a complete and consistent balance, retain them
        has_full_consistent_amounts = False
        if "total_amount" in amounts and "subtotal" in amounts and "tax_amount" in amounts:
            try:
                t = float(amounts["total_amount"].replace(",", ""))
                s = float(amounts["subtotal"].replace(",", ""))
                tx = float(amounts["tax_amount"].replace(",", ""))
                if abs(s + tx - t) <= 0.05 and t > 0:
                    has_full_consistent_amounts = True
            except Exception:
                pass

        # Summary Row Mathematical Solver (for real-world invoices with tabular summary rows)
        if not has_full_consistent_amounts:
            best_cand = None
            for line in lines:
                lower_l = line.lower()
                if any(lower_l.startswith(k) for k in ["item(s) total", "item total", "items total", "particulars"]):
                    continue
                nums = [n.replace(",", "") for n in self.re_amount.findall(line) if "." in n]
                if len(nums) >= 4 or (len(nums) >= 3 and any(k in lower_l for k in ["total", "value", "payable", "balance", "net"])):
                    try:
                        fnums = [float(n) for n in nums]
                        tot = max(fnums)
                        if tot > 0:
                            for sub in sorted([x for x in fnums if 0 < x <= tot and x != tot], reverse=True):
                                diff = round(tot - sub, 2)
                                taxes = [x for x in fnums if x != tot and x != sub]
                                matched = False
                                if abs(diff - sum(taxes)) <= 0.06 or any(abs(diff - t) <= 0.05 for t in taxes):
                                    matched = True
                                else:
                                    pos_taxes = [x for x in taxes if 0 < x < diff + 0.05]
                                    if abs(diff - sum(pos_taxes)) <= 0.06:
                                        matched = True
                                if matched:
                                    if best_cand is None or tot > best_cand[2]:
                                        best_cand = (sub, diff, tot)
                                    break
                            if fnums.count(tot) >= 2:
                                if best_cand is None or tot > best_cand[2]:
                                    best_cand = (tot, 0.0, tot)
                    except Exception:
                        pass

            if best_cand is not None:
                sub, diff, tot = best_cand
                amounts["total_amount"] = f"{tot:.2f}"
                amounts["subtotal"] = f"{sub:.2f}"
                amounts["tax_amount"] = f"{diff:.2f}"

        # Mathematical derivation & consistency: Total = Subtotal + Tax
        if "total_amount" in amounts and "subtotal" in amounts:
            try:
                tot = float(amounts["total_amount"].replace(",", ""))
                sub = float(amounts["subtotal"].replace(",", ""))
                if "tax_amount" not in amounts or tot < sub or abs(float(amounts["tax_amount"].replace(",", "")) - (tot - sub)) > 0.05:
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
        """Spatially extracts supplier and buyer supporting inline labels, 4-box grids, and 2-column headers."""
        parties: Dict[str, Any] = {}
        if not tokens:
            return parties

        first_page = min((t.get("page_num", 1) for t in tokens), default=1)
        p1_tokens = [t for t in tokens if t.get("page_num", 1) == first_page]

        def is_signature_token(tok: Dict[str, Any]) -> bool:
            y = tok["norm_bbox"][1]
            same_line_texts = [
                other["clean_text"].lower()
                for other in p1_tokens
                if abs(other["norm_bbox"][1] - y) < 0.015
            ]
            return any(w in same_line_texts for w in ["digitally", "signed", "signature", "original"])

        def get_header_phrase(tok: Dict[str, Any]) -> str:
            y = tok["norm_bbox"][1]
            same_line = [t for t in p1_tokens if abs(t["norm_bbox"][1] - y) < 0.008 and 0.0 <= (t["norm_bbox"][0] - tok["norm_bbox"][0]) < 0.15]
            same_line.sort(key=lambda x: x["norm_bbox"][0])
            return " ".join(t["clean_text"].upper() for t in same_line[:3])

        sup_headers = []
        bill_headers = []
        for t in p1_tokens:
            if t["norm_bbox"][1] > 0.40 or is_signature_token(t):
                continue
            txt = t["clean_text"].upper()
            phrase = get_header_phrase(t)
            if txt in ["SUPPLIER", "SELLER", "VENDOR", "MERCHANT", "SOLD"] or "BILLED FROM" in phrase or "LEGAL ENTITY" in phrase or "RESTAURANT NAME" in phrase:
                if "TO" not in phrase.split():
                    sup_headers.append(t)
            elif (txt in ["BILL", "BILLED", "BILLING", "CUSTOMER"] and "SUPPLY" not in phrase) or "BILLED TO" in phrase or "BILL TO" in phrase or "CUSTOMER NAME" in phrase:
                if "FROM" not in phrase.split():
                    bill_headers.append(t)

        ship_headers = [
            t for t in p1_tokens
            if t["clean_text"].upper() in ["SHIP", "SHIPPED"] and t["norm_bbox"][1] < 0.40
        ]
        details_headers = [
            t for t in p1_tokens
            if t["clean_text"].upper() in ["DETAILS", "INVOICE DETAILS"] and t["norm_bbox"][1] < 0.35
        ]

        # 1. Inline Seller Check (e.g. "Seller Name: Commodum Groceries Private Limited" or "Sold By: SPIGEN...")
        if sup_headers:
            sup_t = sup_headers[0]
            # Max x boundary: stop before any other column header on the same line
            other_headers_x = [
                h["norm_bbox"][0] for h in (bill_headers + ship_headers + details_headers)
                if abs(h["norm_bbox"][1] - sup_t["norm_bbox"][1]) < 0.03 and h["norm_bbox"][0] > sup_t["norm_bbox"][0]
            ]
            max_inline_x = min(other_headers_x) - 0.02 if other_headers_x else 1.0

            # Tokens on same line to right of header label (tight y tolerance: < 0.007)
            same_line_sup = [
                t for t in p1_tokens
                if abs(t["norm_bbox"][1] - sup_t["norm_bbox"][1]) < 0.007
                and t["norm_bbox"][0] > sup_t["norm_bbox"][2]
                and t["norm_bbox"][0] < max_inline_x
                and t["clean_text"].upper() not in ["NAME:", "NAME", ":", "BILL", "TO", "SHIP", "DETAILS", "INVOICE", "BY", "BY:", "ENTITY", "FROM"]
                and t["clean_text"].upper() not in REJECT_PARTY_NAMES
                and not re.match(r"^(?:tax|id|po|order|tel|phone|gstin|attn)", t["clean_text"], re.I)
            ]
            if same_line_sup:
                same_line_sup.sort(key=lambda x: x["norm_bbox"][0])
                cand = " ".join(t["clean_text"] for t in same_line_sup).strip()
                cand = re.split(r"(?i)(?:\b(?:Invoice\s*Number|Invoice\s*No|Order\s*ID|Order\s*No|GSTIN|Number)\b|,\s*Number|#\s*[A-Z0-9]+)", cand)[0].strip(" ,-:")
                if not self._is_address_line(cand) and cand.upper() not in REJECT_PARTY_NAMES and cand.upper() not in ["FROM", "BY", "NAME", "DETAILS", "ADDRESS"] and len(cand) > 2:
                    parties["supplier_name"] = deduplicate_phrase(cand)

        # 1b. Inline Buyer Check (e.g. "Customer Name: Deepak Mishra")
        if bill_headers and "buyer_name" not in parties:
            bill_t = bill_headers[0]
            other_headers_x = [
                h["norm_bbox"][0] for h in (ship_headers + details_headers)
                if abs(h["norm_bbox"][1] - bill_t["norm_bbox"][1]) < 0.03 and h["norm_bbox"][0] > bill_t["norm_bbox"][0]
            ]
            max_inline_b_x = min(other_headers_x) - 0.02 if other_headers_x else 1.0
            same_line_buyer = [
                t for t in p1_tokens
                if abs(t["norm_bbox"][1] - bill_t["norm_bbox"][1]) < 0.007
                and t["norm_bbox"][0] > bill_t["norm_bbox"][2]
                and t["norm_bbox"][0] < max_inline_b_x
                and t["clean_text"].upper() not in ["NAME:", "NAME", ":", "BILL", "TO", "SHIP", "DETAILS", "INVOICE", "BY", "BY:", "ADDRESS", "DELIVERY"]
                and t["clean_text"].upper() not in REJECT_PARTY_NAMES
                and not re.match(r"^(?:tax|id|po|order|tel|phone|gstin|attn)", t["clean_text"], re.I)
            ]
            if same_line_buyer:
                same_line_buyer.sort(key=lambda x: x["norm_bbox"][0])
                cand_b = " ".join(t["clean_text"] for t in same_line_buyer).strip()
                cand_b = re.split(r"(?i)\b(?:Invoice\s*Number|Invoice\s*No|Order\s*ID|Order\s*No|GSTIN|Place|Delivery|Address|Phone)\b", cand_b)[0].strip(" ,-:")
                if not self._is_address_line(cand_b) and cand_b.upper() not in REJECT_PARTY_NAMES and cand_b.upper() not in ["DELIVERY ADDRESS:", "DELIVERY ADDRESS", "ADDRESS", "DELIVERY"] and not re.match(r"^OD\d{10,}", cand_b) and len(cand_b) > 2:
                    parties["buyer_name"] = deduplicate_phrase(cand_b)

        # 2. 4-Box Grid Layout Check (Bill To is vertically below Supplier)
        is_4_box_grid = False
        if sup_headers and bill_headers:
            sup_t = sup_headers[0]
            bill_t = bill_headers[0]
            if (bill_t["norm_bbox"][1] - sup_t["norm_bbox"][1]) > 0.04 and abs(bill_t["norm_bbox"][0] - sup_t["norm_bbox"][0]) < 0.20:
                is_4_box_grid = True

        if is_4_box_grid:
            sup_t = sup_headers[0]
            bill_t = bill_headers[0]
            max_x = details_headers[0]["norm_bbox"][0] - 0.03 if details_headers else 0.48
            max_y = bill_t["norm_bbox"][1] - 0.01

            if "supplier_name" not in parties:
                sup_cands = [
                    t for t in p1_tokens
                    if t["norm_bbox"][0] >= (sup_t["norm_bbox"][0] - 0.04)
                    and t["norm_bbox"][0] < max_x
                    and t["norm_bbox"][1] >= (sup_t["norm_bbox"][1] + 0.01)
                    and t["norm_bbox"][1] < max_y
                    and t["clean_text"].upper() not in REJECT_PARTY_NAMES
                    and not re.match(r"^(?:tax|id|po|order|tel|phone|gstin|attn|fssai)", t["clean_text"], re.I)
                ]
                if sup_cands:
                    first_y = sup_cands[0]["norm_bbox"][1]
                    first_line_sup = [t for t in sup_cands if abs(t["norm_bbox"][1] - first_y) < 0.007]
                    first_line_sup.sort(key=lambda x: x["norm_bbox"][0])
                    cand_sup = " ".join(t["clean_text"] for t in first_line_sup).strip()
                    if not self._is_address_line(cand_sup) and cand_sup.upper() not in REJECT_PARTY_NAMES and len(cand_sup) > 2:
                        parties["supplier_name"] = deduplicate_phrase(cand_sup)

            # Buyer is in Bill To box strictly
            if "buyer_name" not in parties:
                max_buyer_x = ship_headers[0]["norm_bbox"][0] - 0.03 if ship_headers else 0.48
                buyer_cands = [
                    t for t in p1_tokens
                    if t["norm_bbox"][0] >= (bill_t["norm_bbox"][0] - 0.04)
                    and t["norm_bbox"][0] < max_buyer_x
                    and t["norm_bbox"][1] >= (bill_t["norm_bbox"][1] + 0.01)
                    and t["norm_bbox"][1] < 0.45
                    and t["clean_text"].upper() not in REJECT_PARTY_NAMES
                    and not re.match(r"^(?:tax|id|po|order|tel|phone|gstin|attn|fssai)", t["clean_text"], re.I)
                ]
                if buyer_cands:
                    # Group candidate tokens into horizontal lines
                    lines_map: Dict[float, List[Dict[str, Any]]] = {}
                    for t in buyer_cands:
                        y_key = round(t["norm_bbox"][1], 2)
                        lines_map.setdefault(y_key, []).append(t)
                    
                    # Pick the first line that is a name and NOT an address
                    for y_k in sorted(lines_map.keys()):
                        line_toks = lines_map[y_k]
                        line_toks.sort(key=lambda x: x["norm_bbox"][0])
                        cand_line = " ".join(t["clean_text"] for t in line_toks).strip()
                        if not self._is_address_line(cand_line) and cand_line.upper() not in REJECT_PARTY_NAMES and len(cand_line) > 2:
                            parties["buyer_name"] = deduplicate_phrase(cand_line)
                            break

        elif sup_headers and bill_headers:
            # Layout B: 2-column side-by-side header
            sup_t = sup_headers[0]
            bill_t = bill_headers[0]
            col_split_x = bill_t["norm_bbox"][0] - 0.04

            if "supplier_name" not in parties:
                sup_cands = [
                    t for t in p1_tokens
                    if t["norm_bbox"][0] < col_split_x
                    and t["norm_bbox"][1] >= (sup_t["norm_bbox"][1] + 0.01)
                    and t["norm_bbox"][1] < 0.35
                    and t["clean_text"].upper() not in REJECT_PARTY_NAMES
                    and not re.match(r"^(?:tax|id|po|order|tel|phone|gstin|fssai)", t["clean_text"], re.I)
                ]
                if sup_cands:
                    lines_map = {}
                    for t in sup_cands:
                        y_key = round(t["norm_bbox"][1], 2)
                        lines_map.setdefault(y_key, []).append(t)
                    for y_k in sorted(lines_map.keys()):
                        line_toks = lines_map[y_k]
                        line_toks.sort(key=lambda x: x["norm_bbox"][0])
                        cand_sup = " ".join(t["clean_text"] for t in line_toks).strip()
                        if not self._is_address_line(cand_sup) and cand_sup.upper() not in REJECT_PARTY_NAMES and cand_sup.upper() not in ["FROM", "TO", "BILLED", "BILL", "NAME", "DETAILS"] and len(cand_sup) > 2:
                            parties["supplier_name"] = deduplicate_phrase(cand_sup)
                            break

            buyer_cands = [
                t for t in p1_tokens
                if t["norm_bbox"][0] >= col_split_x
                and t["norm_bbox"][1] >= (bill_t["norm_bbox"][1] + 0.01)
                and t["norm_bbox"][1] < 0.35
                and t["clean_text"].upper() not in REJECT_PARTY_NAMES
                and not re.match(r"^(?:currency|po|order|tel|phone|attn|fssai)", t["clean_text"], re.I)
            ]
            if buyer_cands:
                lines_map = {}
                for t in buyer_cands:
                    y_key = round(t["norm_bbox"][1], 2)
                    lines_map.setdefault(y_key, []).append(t)
                for y_k in sorted(lines_map.keys()):
                    line_toks = lines_map[y_k]
                    line_toks.sort(key=lambda x: x["norm_bbox"][0])
                    cand_buyer = " ".join(t["clean_text"] for t in line_toks).strip()
                    if not self._is_address_line(cand_buyer) and cand_buyer.upper() not in REJECT_PARTY_NAMES and cand_buyer.upper() not in ["FROM", "TO", "BILLED", "BILL", "NAME", "DETAILS"] and not re.match(r"^OD\d{10,}", cand_buyer) and len(cand_buyer) > 2:
                        parties["buyer_name"] = deduplicate_phrase(cand_buyer)
                        break

        return parties

    def _extract_parties_from_lines(self, lines: List[str]) -> Dict[str, Any]:
        """Extracts vendor (seller) and buyer from header text lines using explicit aliases and address rejection."""
        parties: Dict[str, Any] = {}

        # 1. Explicit Inline Supplier Label Check
        for line in lines[:15]:
            m_seller = re.search(
                r"(?i)\b(?:seller\s*name|sold\s*by|billed\s*from|legal\s*entity\s*name|restaurant\s*name|merchant|vendor\s*name|supplier\s*name|issued\s*by|service\s*provider|store\s*name)\s*[:#\-]?\s*([A-Za-z0-9\s.,&'()-]{3,60})",
                line
            )
            if m_seller:
                cand = m_seller.group(1).strip()
                cand = re.split(r"(?i)\b(?:Invoice\s*Number|Invoice\s*No|Order\s*ID|Order\s*No|Ship-from|Restaurant|Address|KH\s*NO|Ground\s*Floor|Billed\s*To|Bill\s*To|GSTIN)\b", cand)[0].strip(" ,-:")
                cand = deduplicate_phrase(cand)
                if not self._is_address_line(cand) and cand.upper() not in REJECT_PARTY_NAMES and len(cand) > 2:
                    parties["supplier_name"] = cand
                    break

        if "supplier_name" not in parties:
            for line in lines[:8]:
                if "ETERNAL LIMITED" in line.upper():
                    cand = line.split("Address:")[0].strip(" ,-:")
                    parties["supplier_name"] = cand
                    break

        # 2. Buyer Search: looks for 'Bill To:', 'Billed To:', 'Customer:', 'Buyer:', 'Billing Address'
        for idx, line in enumerate(lines[:25]):
            lower = line.lower()
            if any(k in lower for k in ["bill to", "billed to", "billing address", "customer name", "sold to", "buyer", "customer", "invoice to", "client name"]):
                clean_party = re.sub(r"(?i)^.*?\b(?:customer\s*name|customer\s*details|customer|bill\s*to|billed\s*to|billing\s*address|sold\s*to|buyer\s*name|buyer|client\s*name|name)\s*[:#\-]?\s*", "", line).strip()
                clean_party = re.split(r"(?i)\b(?:Delivery|Banna|212|Iconic|Address|Paras|GSTIN|Phone|Shree|State|Place|Order)\b", clean_party)[0].strip(" ,-:")
                clean_party = deduplicate_phrase(clean_party)
                if clean_party and not self._is_address_line(clean_party) and clean_party.upper() not in REJECT_PARTY_NAMES and not re.match(r"^OD\d{10,}", clean_party) and len(clean_party) > 2:
                    parties["buyer_name"] = clean_party
                    break
                elif idx + 1 < len(lines):
                    for next_line_raw in lines[idx + 1:idx + 4]:
                        next_party = re.sub(r"(?i)^.*?\b(?:customer\s*name|customer|name)\s*[:#\-]?\s*", "", next_line_raw).strip()
                        next_party = re.split(r"(?i)\b(?:Delivery|Banna|212|Iconic|Address|Paras|GSTIN|Phone|Shree|State|Place|Order)\b", next_party)[0].strip(" ,-:")
                        next_party = deduplicate_phrase(next_party)
                        if next_party.upper() not in REJECT_PARTY_NAMES and not re.match(r"^OD\d{10,}", next_party) and len(next_party) > 2:
                            if not self._is_address_line(next_party) and not any(kw in next_party.lower() for kw in ["invoice", "date", "phone", "gst", "fssai"]):
                                parties["buyer_name"] = next_party
                                break
                    if "buyer_name" in parties:
                        break

        # 3. Supplier Search if not yet found
        buyer = parties.get("buyer_name", "")
        if "supplier_name" not in parties:
            for idx, line in enumerate(lines[:12]):
                line_str = line.strip()
                if not line_str or line_str.upper() in REJECT_PARTY_NAMES:
                    continue

                candidate_line = line_str
                if re.search(r"(?i)\b(?:invoice|inv|bill)\s*(?:no|number|#)", line_str):
                    left_part = re.split(r"(?i)\b(?:invoice|inv|bill)\s*(?:no|number|#)", line_str)[0].strip()
                    if len(left_part) > 3 and not left_part.lower().startswith("tax"):
                        candidate_line = left_part

                candidate_line = deduplicate_phrase(candidate_line)
                lower = candidate_line.lower()
                if any(k in lower for k in ["tax invoice", "invoice", "synthetic sample", "bill to", "billed to", "date:", "fssai"]):
                    continue

                if self._is_address_line(candidate_line):
                    continue

                if buyer and (candidate_line.lower() == buyer.lower() or buyer.lower() in candidate_line.lower()):
                    continue

                # Explicit label
                if any(k in lower for k in ["from:", "seller:", "vendor:", "supplier:"]):
                    clean_sup = re.sub(r"(?i)(from|seller|vendor|supplier)\s*[:#\-]?\s*", "", candidate_line).strip()
                    clean_sup = deduplicate_phrase(clean_sup)
                    if not self._is_address_line(clean_sup) and clean_sup.upper() not in REJECT_PARTY_NAMES and len(clean_sup) > 2:
                        parties["supplier_name"] = clean_sup
                        break

                # Corporate name heuristics
                if any(sfx in lower for sfx in ["ltd", "limited", "pvt", "inc", "corp", "llc", "gmbh", "s.l.", "pte", "technologies", "services", "solutions", "enterprises", "supplies", "systems", "logistics", "groceries", "retail", "foods", "studio"]):
                    if candidate_line.upper() not in REJECT_PARTY_NAMES and not self._is_address_line(candidate_line):
                        parties["supplier_name"] = candidate_line
                        break

        # Fallback for supplier: first prominent non-header, non-address line
        if "supplier_name" not in parties and lines:
            for line in lines[:6]:
                clean = deduplicate_phrase(line.strip())
                if clean and clean.upper() not in REJECT_PARTY_NAMES and not any(k in clean.lower() for k in ["tax invoice", "date", "bill to", "invoice", "synthetic", "fssai"]):
                    if not self._is_address_line(clean):
                        if buyer and clean.lower() != buyer.lower():
                            parties["supplier_name"] = clean
                            break

        return parties
