"""
Line-Item Table Extraction Module.
Uses spatial OCR coordinates, column boundary detection, horizontal row clustering,
and mathematical constraint verification to extract invoice line items.
"""

from typing import List, Dict, Any, Optional, Tuple
import re


class LineItemExtractor:
    """Extracts tabular line items (description, quantity, unit price, tax, total) using geometry and rules."""

    def __init__(self):
        # Column header keywords
        self.col_keywords = {
            "description": ["description", "item", "particulars", "product", "details", "service", "desc"],
            "quantity": ["qty", "quantity", "qnty", "units", "count", "hrs", "hours"],
            "unit_price": ["rate", "price", "unit price", "unit_price", "unit cost", "cost"],
            "tax": ["tax", "tax %", "tax rate", "gst", "vat", "cgst", "sgst", "igst"],
            "total": ["amount", "total", "line total", "net amount", "total amt", "line total incl. tax"],
        }
        self.table_end_keywords = [
            "subtotal", "sub total", "sub-total", "taxable amount",
            "total tax", "grand total", "net amount", "terms & conditions",
            "bank details", "signature", "authorized", "thank you",
        ]
        self.num_regex = re.compile(r"^\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)$")

    def extract_line_items(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts structured line items from document tokens.
        """
        if not tokens:
            return []

        # 1. Group tokens into horizontal lines
        lines = self._group_tokens_by_y(tokens)
        if not lines:
            return []

        # 2. Locate table header line and determine column bounding spans
        header_idx, col_spans = self._detect_table_headers(lines)
        if header_idx is None or not col_spans:
            # Fallback to coordinate-less line scanning
            return self._fallback_line_item_parser(lines)

        # 3. Find end of table (subtotal or total line)
        end_idx = self._detect_table_end(lines, header_idx)

        # 4. Extract rows between header and end
        raw_rows = lines[header_idx + 1:end_idx]
        extracted_items: List[Dict[str, Any]] = []

        for row in raw_rows:
            item = self._parse_row_with_columns(row, col_spans)
            if item and self._is_valid_item(item):
                extracted_items.append(item)

        # If strict column mapping yielded nothing, try heuristic line parsing
        if not extracted_items:
            extracted_items = self._fallback_line_item_parser(raw_rows)

        return extracted_items

    def _group_tokens_by_y(self, tokens: List[Dict[str, Any]], y_thresh: float = 0.012) -> List[List[Dict[str, Any]]]:
        """Clusters tokens into lines by vertical coordinate."""
        sorted_toks = sorted(tokens, key=lambda t: (t.get("page_num", 1), t["norm_bbox"][1], t["norm_bbox"][0]))
        lines: List[List[Dict[str, Any]]] = []
        cur_line: List[Dict[str, Any]] = []
        cur_page = None
        cur_y = None

        for t in sorted_toks:
            p = t.get("page_num", 1)
            y = t["norm_bbox"][1]

            if cur_page is None:
                cur_page = p
                cur_y = y
                cur_line = [t]
            elif p != cur_page or abs(y - cur_y) > y_thresh:
                if cur_line:
                    cur_line.sort(key=lambda tok: tok["norm_bbox"][0])
                    lines.append(cur_line)
                cur_page = p
                cur_y = y
                cur_line = [t]
            else:
                cur_line.append(t)

        if cur_line:
            cur_line.sort(key=lambda tok: tok["norm_bbox"][0])
            lines.append(cur_line)

        return lines

    def _detect_table_headers(self, lines: List[List[Dict[str, Any]]]) -> Tuple[Optional[int], Dict[str, float]]:
        """Identifies line index with the most table header keywords and determines column x-positions."""
        best_idx = None
        best_score = 0
        best_col_positions: Dict[str, float] = {}

        for idx, line in enumerate(lines[:30]):
            line_text = " ".join(t["clean_text"].lower() for t in line)
            matched_cols = {}

            for col, kws in self.col_keywords.items():
                for t in line:
                    word = t["clean_text"].lower().strip(":;#")
                    if any(kw == word or kw in word for kw in kws):
                        matched_cols[col] = t["norm_bbox"][0]
                        break

            # Need at least 2 distinct column indicators (e.g. description and amount/qty)
            if len(matched_cols) >= 2 and len(matched_cols) > best_score:
                best_score = len(matched_cols)
                best_idx = idx
                best_col_positions = matched_cols

        return best_idx, best_col_positions

    def _detect_table_end(self, lines: List[List[Dict[str, Any]]], start_idx: int) -> int:
        """Finds line where the item table terminates."""
        for idx in range(start_idx + 1, len(lines)):
            line_text = " ".join(t["clean_text"].lower() for t in lines[idx])
            if any(term in line_text for term in self.table_end_keywords):
                return idx
        return len(lines)

    def _parse_row_with_columns(
        self, row: List[Dict[str, Any]], col_spans: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Maps tokens in a single row to closest detected column headers using spatial column boundaries."""
        currencies = {
            "USD", "EUR", "INR", "GBP", "CAD", "SGD", "AUD", "JPY", "CHF", "CNY",
            "$", "€", "£", "₹", "RS", "RS."
        }

        item: Dict[str, Any] = {
            "description": "",
            "quantity": None,
            "unit_price": None,
            "tax_rate": None,
            "tax_amount": None,
            "total": None,
        }

        # Retrieve column boundary positions
        q_pos = col_spans.get("quantity")
        p_pos = col_spans.get("unit_price")
        tax_pos = col_spans.get("tax")
        tot_pos = col_spans.get("total")

        desc_tokens = []
        numbers_with_x = []

        # First column after description
        first_num_col_pos = min([pos for pos in [q_pos, p_pos, tax_pos, tot_pos] if pos is not None], default=0.45)

        for t in row:
            text = t["clean_text"]
            x = t["norm_bbox"][0]
            clean_num = text.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("₹", "")

            # If token is in description column (left of first numerical column)
            if x < (first_num_col_pos - 0.02):
                if text.upper() not in currencies and not self._is_float(clean_num):
                    desc_tokens.append(text)
                elif self._is_float(clean_num) and desc_tokens:
                    # Allow numbers inside descriptions (e.g. "A4 Paper", "Item 10")
                    desc_tokens.append(text)
            else:
                if self._is_float(clean_num):
                    val = float(clean_num)
                    numbers_with_x.append((x, val))

        raw_desc = " ".join(desc_tokens).strip()
        # Clean currency codes and punctuation artifact from description
        clean_desc = re.sub(r"\b(?:USD|EUR|INR|GBP|CAD|SGD|AUD|JPY|CHF|CNY)\b", "", raw_desc).strip()
        clean_desc = re.sub(r"\s+", " ", clean_desc).strip(" -:;,")
        item["description"] = clean_desc

        if numbers_with_x:
            numbers_with_x.sort(key=lambda n: n[0])
            num_vals = [n[1] for n in numbers_with_x]

            # Direct mapping for standard multi-column tables
            if len(num_vals) == 4 and tax_pos is not None:
                item["quantity"] = num_vals[0]
                item["unit_price"] = num_vals[1]
                item["tax_amount"] = num_vals[2]
                item["total"] = num_vals[3]
            elif len(num_vals) == 3 and tax_pos is None:
                item["quantity"] = num_vals[0]
                item["unit_price"] = num_vals[1]
                item["total"] = num_vals[2]
            else:
                # Try to assign by spatial column proximity
                assigned_indices = set()

                if q_pos is not None:
                    # Find closest number to q_pos
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - q_pos))
                    if abs(numbers_with_x[cand_idx][0] - q_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["quantity"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

                if tot_pos is not None:
                    # Total is typically furthest right or closest to tot_pos
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - tot_pos))
                    if abs(numbers_with_x[cand_idx][0] - tot_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["total"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

                if p_pos is not None:
                    # Unit price
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - p_pos))
                    if abs(numbers_with_x[cand_idx][0] - p_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["unit_price"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

                if tax_pos is not None:
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - tax_pos))
                    if abs(numbers_with_x[cand_idx][0] - tax_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["tax_amount"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

                # Fallback to standard left-to-right order for unassigned items
                if item["total"] is None and num_vals:
                    item["total"] = num_vals[-1]

                if item["quantity"] is None:
                    if len(num_vals) >= 2:
                        item["quantity"] = num_vals[0]
                    elif len(num_vals) == 1:
                        item["quantity"] = 1.0

                if item["unit_price"] is None:
                    if len(num_vals) >= 3:
                        item["unit_price"] = num_vals[1]
                    elif item["total"] is not None and item["quantity"] is not None and item["quantity"] > 0:
                        item["unit_price"] = round(item["total"] / item["quantity"], 2)

        return item

    def _fallback_line_item_parser(self, lines: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fallback pattern-based line item detector for unaligned or unusual tables."""
        items: List[Dict[str, Any]] = []

        for row in lines:
            row_text = " ".join(t["clean_text"] for t in row).strip()
            if not row_text:
                continue

            # Check if line contains numbers at the end
            words = row_text.split()
            if len(words) < 2:
                continue

            # Check table end words
            if any(k in row_text.lower() for k in self.table_end_keywords):
                continue

            # Look for amounts at the end of the line
            numeric_indices = []
            for idx, w in enumerate(words):
                clean = w.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("₹", "")
                if self._is_float(clean):
                    numeric_indices.append((idx, float(clean)))

            if numeric_indices:
                first_num_idx = numeric_indices[0][0]
                desc = " ".join(words[:first_num_idx]).strip()
                if not desc or len(desc) < 2:
                    continue

                vals = [n[1] for n in numeric_indices]
                item = {
                    "description": desc,
                    "quantity": vals[0] if len(vals) >= 2 else 1.0,
                    "unit_price": vals[1] if len(vals) >= 3 else (vals[0] if len(vals) == 2 else vals[0]),
                    "tax_rate": None,
                    "tax_amount": None,
                    "total": vals[-1],
                }
                if self._is_valid_item(item):
                    items.append(item)

        return items

    @staticmethod
    def _is_float(val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_valid_item(item: Dict[str, Any]) -> bool:
        """Validates that extracted item has substantive content and reasonable values."""
        desc = item.get("description", "")
        total = item.get("total")
        # Must have non-trivial description and at least a total or price
        if not desc or len(desc) < 2:
            return False
        if total is None and item.get("unit_price") is None:
            return False
        # Ignore lines that are headers or summary lines
        lower_desc = desc.lower()
        if any(h in lower_desc for h in ["subtotal", "total", "tax", "date", "invoice", "page"]):
            return False
        return True
