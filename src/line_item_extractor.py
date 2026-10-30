"""
Line-Item Table Extraction Module.
Uses spatial OCR coordinates, multi-line header clustering, column boundary detection,
horizontal row grouping, serial number anchoring, and mathematical constraint verification
to extract invoice line items.
"""

from typing import List, Dict, Any, Optional, Tuple
import re


class LineItemExtractor:
    """Extracts tabular line items using geometry, column intervals, and serial-number anchoring."""

    def __init__(self):
        # Column header keywords
        self.col_keywords = {
            "sr_no": ["sr", "sl", "s.no", "sr no", "sl no", "s no", "sl.", "item no", "no"],
            "description": ["description", "item", "particulars", "product", "details", "service", "desc", "title", "product title"],
            "mrp": ["mrp", "rsp", "mrp/rsp", "unit mrp"],
            "hsn": ["hsn", "sac", "hsn/sac", "hsn code"],
            "quantity": ["qty", "quantity", "qnty", "units", "count", "hrs", "hours"],
            "unit_price": ["unit price", "unit_price", "product rate", "rate", "price", "unit cost", "cost"],
            "gross_amount": ["gross", "gross amount", "gross value"],
            "discount": ["discount", "disc", "disc."],
            "taxable_amount": ["taxable", "taxable amt", "taxable value", "taxable amount", "assessable", "net value", "net amount"],
            "cgst_amt": ["cgst amt", "cgst (inr)", "cgst amount"],
            "sgst_amt": ["sgst amt", "sgst (inr)", "sgst amount", "sgst/utgst", "utgst"],
            "igst_amt": ["igst", "igst amt", "igst (inr)", "igst amount"],
            "tax": ["tax", "tax %", "tax rate", "gst", "vat", "cgst", "sgst", "igst", "cess"],
            "total": ["total amount", "total amt", "line total", "total value", "invoice value", "item total", "total"],
        }
        self.table_end_keywords = [
            "subtotal", "sub total", "sub-total", "taxable amount",
            "total tax", "grand total", "net amount", "total amount",
            "item total", "invoice value", "total value", "total:",
            "amount in words", "details of goods transported", "gta supplier",
            "terms & conditions", "bank details", "signature", "authorized", "thank you",
            "whether tax is payable", "whether gst is payable", "order delivered from",
            "fssai", "fbo",
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
        header_start_idx, header_end_idx, col_spans = self._detect_table_headers(lines)
        if header_start_idx is None or not col_spans:
            # Fallback to coordinate-less line scanning
            return self._fallback_line_item_parser(lines)

        # 3. Find end of table (subtotal or total line)
        end_idx = self._detect_table_end(lines, header_end_idx)

        # Restrict tokens to same page as table header
        header_page = lines[header_start_idx][0].get("page_num", 1)
        raw_rows = [
            line for line in lines[header_end_idx + 1:end_idx]
            if line[0].get("page_num", 1) == header_page
        ]

        if not raw_rows:
            return []

        # 4. Check for serial-number anchored multi-line rows
        sr_items = self._try_extract_sr_anchored_items(raw_rows, lines, header_end_idx, end_idx, col_spans, tokens, header_page)
        if sr_items:
            return sr_items

        # 5. Standard single-line or multi-line row extraction
        extracted_items: List[Dict[str, Any]] = []
        cur_multiline_desc = []

        for row in raw_rows:
            # Check if row is purely summary row or footer
            row_text = " ".join(t["clean_text"].lower() for t in row)
            if any(k in row_text for k in self.table_end_keywords):
                break

            item, row_desc_only = self._parse_row_with_columns(row, col_spans)
            if row_desc_only:
                cur_multiline_desc.append(row_desc_only)
                continue

            if item and self._is_valid_item(item):
                if cur_multiline_desc:
                    item["description"] = " ".join(cur_multiline_desc) + " " + item["description"]
                    item["description"] = self._clean_item_description(item["description"])
                    cur_multiline_desc = []
                extracted_items.append(item)

        # If strict column mapping yielded nothing, try heuristic line parsing
        if not extracted_items:
            extracted_items = self._fallback_line_item_parser(raw_rows)

        return extracted_items

    def _group_tokens_by_y(self, tokens: List[Dict[str, Any]], y_thresh: float = 0.007) -> List[List[Dict[str, Any]]]:
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

    def _detect_table_headers(self, lines: List[List[Dict[str, Any]]]) -> Tuple[Optional[int], Optional[int], Dict[str, float]]:
        """Identifies header line(s) and determines column x-positions."""
        best_start = None
        best_end = None
        best_spans: Dict[str, float] = {}

        # Scan for potential header blocks in the first 35 lines
        for start_idx in range(min(35, len(lines))):
            col_spans: Dict[str, float] = {}
            end_idx = start_idx

            # Collect headers over up to 5 consecutive lines (multi-line table headers)
            for offset in range(5):
                cur_idx = start_idx + offset
                if cur_idx >= len(lines):
                    break
                line = lines[cur_idx]
                
                # A header line must not contain data numbers/amounts
                num_count = sum(
                    1 for t in line
                    if self._is_float(t["clean_text"].replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("₹", ""))
                )
                if num_count > 0:
                    break

                line_matched = 0

                for t in line:
                    word = t["clean_text"].lower().strip(":;#")
                    x = t["norm_bbox"][0]
                    for col_name, kws in self.col_keywords.items():
                        if any(kw == word for kw in kws):
                            if col_name not in col_spans:
                                col_spans[col_name] = x
                            line_matched += 1
                            break

                if line_matched > 0:
                    end_idx = cur_idx
                elif offset > 0:
                    break

            # Must have at least description and one numeric column (e.g. total or qty)
            has_desc = "description" in col_spans
            has_num = any(k in col_spans for k in ["total", "quantity", "unit_price", "taxable_amount"])

            if has_desc and has_num and len(col_spans) > len(best_spans):
                best_start = start_idx
                best_end = end_idx
                best_spans = col_spans

        return best_start, best_end, best_spans

    def _detect_table_end(self, lines: List[List[Dict[str, Any]]], start_idx: int) -> int:
        """Finds line where the item table terminates."""
        header_page = lines[start_idx][0].get("page_num", 1)

        for idx in range(start_idx + 1, len(lines)):
            line = lines[idx]
            # Stop if page advances
            if line[0].get("page_num", 1) != header_page:
                return idx

            line_text = " ".join(t["clean_text"].lower() for t in line)
            if any(term in line_text for term in self.table_end_keywords):
                return idx

        return len(lines)

    def _try_extract_sr_anchored_items(
        self,
        raw_rows: List[List[Dict[str, Any]]],
        all_lines: List[List[Dict[str, Any]]],
        header_end_idx: int,
        end_idx: int,
        col_spans: Dict[str, float],
        all_tokens: List[Dict[str, Any]],
        header_page: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extracts items when table uses explicit sequential serial numbers (1, 2, 3...)
        and wrapped multi-line descriptions.
        """
        sr_x = col_spans.get("sr_no", 0.08)
        desc_x = col_spans.get("description", 0.12)
        # Numerical columns start after description
        num_col_xs = [v for k, v in col_spans.items() if k not in ("sr_no", "description")]
        first_num_x = min(num_col_xs) if num_col_xs else 0.25

        # Find serial number anchored rows
        sr_entries: List[Tuple[int, int, float, List[Dict[str, Any]]]] = []
        expected_next = 1

        for line_idx, line in enumerate(raw_rows):
            first_tok = line[0]
            # Must be in serial number column (x < first_num_x and near sr_x)
            if first_tok["norm_bbox"][0] < min(first_num_x, 0.15):
                txt = first_tok["clean_text"].strip(".")
                if txt.isdigit():
                    num = int(txt)
                    if num == expected_next:
                        sr_entries.append((num, line_idx, first_tok["norm_bbox"][1], line))
                        expected_next += 1

        # Require at least 2 sequential serial numbers or 1 with strong table structure
        if len(sr_entries) < 1:
            return None

        # Determine vertical boundaries for each item
        header_bottom = all_lines[header_end_idx][0]["norm_bbox"][3]
        table_bottom = all_lines[end_idx - 1][-1]["norm_bbox"][3] if end_idx <= len(all_lines) else 1.0

        # Check if there is a summary row before table_end (e.g. total row with no SR)
        last_sr_line_idx = sr_entries[-1][1]
        for post_idx in range(last_sr_line_idx + 1, len(raw_rows)):
            post_line = raw_rows[post_idx]
            post_text = " ".join(t["clean_text"].lower() for t in post_line)
            # If line has multiple numbers across columns but no description, it's a summary row
            num_tokens = [t for t in post_line if self._is_float(t["clean_text"].replace(",", "").replace("%", ""))]
            if len(num_tokens) >= 3 or any(k in post_text for k in self.table_end_keywords):
                table_bottom = post_line[0]["norm_bbox"][1]
                break

        boundaries = [header_bottom]
        for k in range(len(sr_entries) - 1):
            mid_y = (sr_entries[k][2] + sr_entries[k + 1][2]) / 2.0
            boundaries.append(mid_y)
        boundaries.append(table_bottom)

        extracted_items: List[Dict[str, Any]] = []

        for idx in range(len(sr_entries)):
            sr_num, _, _, num_line = sr_entries[idx]
            top_y = boundaries[idx]
            bot_y = boundaries[idx + 1]

            # Collect description tokens in this vertical slice on header_page
            item_desc_tokens = [
                t for t in all_tokens
                if t.get("page_num", 1) == header_page
                and top_y <= t["norm_bbox"][1] < bot_y
                and (sr_x + 0.01) <= t["norm_bbox"][0] < (first_num_x - 0.01)
            ]

            # Sort description tokens vertically (y coordinate)
            item_desc_tokens.sort(key=lambda t: (t["norm_bbox"][1], t["norm_bbox"][0]))
            raw_desc = " ".join(t["clean_text"] for t in item_desc_tokens)
            desc = self._clean_item_description(raw_desc)

            # Map numbers from the SR row
            item: Dict[str, Any] = {
                "description": desc,
                "quantity": 1.0,
                "unit_price": None,
                "tax_rate": None,
                "taxable_amount": None,
                "tax_amount": None,
                "total": None,
            }

            # Map each token on num_line by proximity to col_spans
            cgst_val = 0.0
            sgst_val = 0.0
            tax_rate_vals = []

            for tok in num_line:
                raw_tok = tok["clean_text"]
                is_pct = "%" in raw_tok
                t_text = raw_tok.replace(",", "").replace("$", "").replace("₹", "").replace("%", "")
                t_x = tok["norm_bbox"][0]

                if not self._is_float(t_text):
                    continue

                val = float(t_text)

                if is_pct:
                    tax_rate_vals.append(val)
                    continue

                # Find closest column
                closest_col = min(col_spans.keys(), key=lambda c: abs(col_spans[c] - t_x))
                dist = abs(col_spans[closest_col] - t_x)

                if dist < 0.09:
                    if closest_col == "quantity":
                        item["quantity"] = val
                    elif closest_col == "unit_price":
                        item["unit_price"] = val
                    elif closest_col == "taxable_amount":
                        item["taxable_amount"] = val
                    elif closest_col == "total":
                        item["total"] = val
                    elif closest_col == "cgst_amt":
                        cgst_val = val
                    elif closest_col == "sgst_amt":
                        sgst_val = val

            # Set tax rate if found
            if tax_rate_vals:
                item["tax_rate"] = tax_rate_vals[-1]

            # Total tax if CGST and SGST columns were present
            if cgst_val > 0 or sgst_val > 0:
                item["tax_amount"] = round(cgst_val + sgst_val, 2)

            # Fallbacks and cross-derivations
            if item["total"] is None and item.get("taxable_amount") is not None:
                item["total"] = round(item["taxable_amount"] + (item.get("tax_amount") or 0.0), 2)
            elif item["total"] is not None and item.get("taxable_amount") is not None and item.get("tax_amount") is None:
                calc_tax = round(item["total"] - item["taxable_amount"], 2)
                if 0.0 < calc_tax < item["total"]:
                    item["tax_amount"] = calc_tax

            if item["unit_price"] is None:
                if item.get("taxable_amount") is not None and item["quantity"] > 0:
                    item["unit_price"] = round(item["taxable_amount"] / item["quantity"], 2)
                elif item["total"] is not None and item["quantity"] > 0:
                    item["unit_price"] = round(item["total"] / item["quantity"], 2)

            if desc and (item["total"] is not None or item["unit_price"] is not None):
                extracted_items.append(item)

        return extracted_items if len(extracted_items) == len(sr_entries) else None

    def _parse_row_with_columns(
        self, row: List[Dict[str, Any]], col_spans: Dict[str, float]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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
            "taxable_amount": None,
            "tax_amount": None,
            "total": None,
        }

        q_pos = col_spans.get("quantity")
        p_pos = col_spans.get("unit_price")
        tax_pos = col_spans.get("tax")
        tot_pos = col_spans.get("total")
        taxable_pos = col_spans.get("taxable_amount")

        desc_tokens = []
        numbers_with_x = []

        first_num_col_pos = min(
            [pos for pos in [q_pos, p_pos, tax_pos, tot_pos, taxable_pos] if pos is not None],
            default=0.45,
        )

        for t in row:
            text = t["clean_text"]
            x = t["norm_bbox"][0]
            clean_num = text.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("₹", "")

            if x < (first_num_col_pos - 0.02):
                if text.upper() not in currencies and not self._is_float(clean_num):
                    desc_tokens.append(text)
                elif self._is_float(clean_num) and desc_tokens:
                    desc_tokens.append(text)
            else:
                if self._is_float(clean_num):
                    val = float(clean_num)
                    numbers_with_x.append((x, val))

        raw_desc = " ".join(desc_tokens).strip()
        cleaned_desc = self._clean_item_description(raw_desc)

        # If line contains only description and zero numbers, it's a wrapped description line
        if not numbers_with_x and cleaned_desc:
            return None, cleaned_desc

        item["description"] = cleaned_desc

        if numbers_with_x:
            numbers_with_x.sort(key=lambda n: n[0])
            num_vals = [n[1] for n in numbers_with_x]

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
                assigned_indices = set()

                if q_pos is not None:
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - q_pos))
                    if abs(numbers_with_x[cand_idx][0] - q_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["quantity"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

                if tot_pos is not None:
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - tot_pos))
                    if abs(numbers_with_x[cand_idx][0] - tot_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["total"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

                if p_pos is not None:
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - p_pos))
                    if abs(numbers_with_x[cand_idx][0] - p_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["unit_price"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

                if tax_pos is not None:
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - tax_pos))
                    if abs(numbers_with_x[cand_idx][0] - tax_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["tax_amount"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

                if taxable_pos is not None:
                    cand_idx = min(range(len(numbers_with_x)), key=lambda i: abs(numbers_with_x[i][0] - taxable_pos))
                    if abs(numbers_with_x[cand_idx][0] - taxable_pos) < 0.12 and cand_idx not in assigned_indices:
                        item["taxable_amount"] = numbers_with_x[cand_idx][1]
                        assigned_indices.add(cand_idx)

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

        return item, None

    def _fallback_line_item_parser(self, lines: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fallback pattern-based line item detector for unaligned or unusual tables."""
        items: List[Dict[str, Any]] = []

        for row in lines:
            row_text = " ".join(t["clean_text"] for t in row).strip()
            if not row_text:
                continue

            words = row_text.split()
            if len(words) < 2:
                continue

            if any(k in row_text.lower() for k in self.table_end_keywords):
                continue

            numeric_indices = []
            for idx, w in enumerate(words):
                clean = w.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("₹", "")
                if self._is_float(clean):
                    numeric_indices.append((idx, float(clean)))

            if numeric_indices:
                first_num_idx = numeric_indices[0][0]
                raw_desc = " ".join(words[:first_num_idx]).strip()
                desc = self._clean_item_description(raw_desc)
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
    def _clean_item_description(desc: str) -> str:
        """Strips currency column artifacts while preserving words and numbers."""
        if not desc:
            return ""
        s = re.sub(r"(?i)\b(?:USD|EUR|INR|GBP|CAD|SGD|AUD|JPY|CHF|CNY|RS|RS\.)\b", " ", desc)
        s = re.sub(r"[\$€£₹¥]", " ", s)
        # Fix vertical pipe spacing e.g. "|Coconut" -> "| Coconut"
        s = re.sub(r"\|(?=[A-Za-z0-9])", "| ", s)
        s = re.sub(r"(?<=[A-Za-z0-9])\|", " |", s)
        s = re.sub(r"\s+", " ", s).strip(" -:;,\t")
        return s

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
        if not desc or len(desc) < 2:
            return False
        if total is None and item.get("unit_price") is None:
            return False
        lower_desc = desc.lower()
        if any(h in lower_desc for h in ["subtotal", "total", "tax", "date", "invoice", "page"]):
            return False
        return True
