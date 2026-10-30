"""
Validation and Quality Assurance Module for Extracted Invoice Data.
Performs schema validation, GSTIN validation, date parsing, numeric conversions,
and mathematical consistency checks (subtotal + tax == total).
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import re
from dateutil import parser as date_parser

from src.config import PATTERNS, CURRENCY_MAP, MATH_TOLERANCE


class InvoiceValidator:
    """Validates extracted invoice fields and asserts financial consistency."""

    def __init__(self, tolerance: float = MATH_TOLERANCE):
        self.tolerance = tolerance
        self.gstin_pattern = re.compile(PATTERNS["gstin"])

    def validate(self, fields: Dict[str, Any], line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates extracted invoice dictionary and line items.
        Returns a dictionary with validation status, normalized fields, warnings, and errors.
        """
        errors: List[str] = []
        warnings: List[str] = []
        cleaned: Dict[str, Any] = dict(fields)

        # 1. Mandatory Field Checks
        if not cleaned.get("invoice_number"):
            warnings.append("Missing 'invoice_number': Could not identify invoice identifier.")
        if not cleaned.get("invoice_date"):
            warnings.append("Missing 'invoice_date': Could not identify issue date.")
        if not cleaned.get("total_amount"):
            warnings.append("Missing 'total_amount': Total invoice value is absent.")
        if not cleaned.get("supplier_name"):
            warnings.append("Missing 'supplier_name': Vendor or seller was not detected.")

        # 2. Date Normalization & Validation
        if cleaned.get("invoice_date"):
            normalized_date, date_err = self._validate_and_normalize_date(str(cleaned["invoice_date"]))
            if date_err:
                warnings.append(f"Invalid date format for '{cleaned['invoice_date']}': {date_err}")
            else:
                cleaned["invoice_date"] = normalized_date

        # 3. Currency Normalization
        if cleaned.get("currency"):
            curr_str = str(cleaned["currency"]).strip().upper()
            curr_str = re.sub(r"[^A-Z\$€£₹¥]", "", curr_str)
            cleaned["currency"] = CURRENCY_MAP.get(curr_str, curr_str)
        else:
            warnings.append("Missing 'currency': Currency symbol or code was not identified.")

        # 4. Tax ID / GSTIN Validation
        if cleaned.get("tax_id"):
            tax_id_str = str(cleaned["tax_id"]).strip().upper()
            if self.gstin_pattern.search(tax_id_str):
                cleaned["tax_id"] = tax_id_str
                cleaned["tax_id_type"] = "GSTIN"
            else:
                cleaned["tax_id_type"] = "TAX_ID"
                if len(tax_id_str) < 5:
                    warnings.append(f"Suspicious Tax ID format: '{tax_id_str}' appears unusually short.")

        # 5. Financial Numeric Cleaning
        subtotal_val = self._parse_numeric(cleaned.get("subtotal"))
        tax_val = self._parse_numeric(cleaned.get("tax_amount"))
        total_val = self._parse_numeric(cleaned.get("total_amount"))

        # Reconcile missing amount if 2 of 3 amounts are present
        if subtotal_val is not None and total_val is not None and tax_val is None:
            if total_val >= subtotal_val:
                tax_val = round(total_val - subtotal_val, 2)
        elif total_val is not None and tax_val is not None and subtotal_val is None:
            if total_val >= tax_val:
                subtotal_val = round(total_val - tax_val, 2)
        elif subtotal_val is not None and tax_val is not None and total_val is None:
            total_val = round(subtotal_val + tax_val, 2)

        # If subtotal is still missing but line items exist
        if subtotal_val is None and line_items:
            items_tot = sum(self._parse_numeric(it.get("total")) or 0.0 for it in line_items)
            if items_tot > 0:
                subtotal_val = round(items_tot, 2)
                if total_val is None and tax_val is not None:
                    total_val = round(subtotal_val + tax_val, 2)
                elif total_val is not None and tax_val is None and total_val >= subtotal_val:
                    tax_val = round(total_val - subtotal_val, 2)

        cleaned["subtotal"] = subtotal_val
        cleaned["tax_amount"] = tax_val
        cleaned["total_amount"] = total_val

        # 6. Mathematical Consistency Check: subtotal + tax_amount == total_amount
        if subtotal_val is not None and tax_val is not None and total_val is not None:
            expected_total = subtotal_val + tax_val
            diff = abs(expected_total - total_val)
            if diff > self.tolerance:
                warnings.append(
                    f"Mathematical Discrepancy: Subtotal ({subtotal_val:.2f}) + Tax ({tax_val:.2f}) "
                    f"= {expected_total:.2f}, but Total is {total_val:.2f} (diff: {diff:.2f})."
                )
        elif subtotal_val is not None and total_val is not None and tax_val is None:
            if total_val < subtotal_val:
                warnings.append(
                    f"Logical Discrepancy: Total amount ({total_val:.2f}) is strictly less than subtotal ({subtotal_val:.2f})."
                )

        # 7. Line Item Validation & Cross-Reconciliation
        cleaned_items, item_warnings = self._validate_line_items(line_items)
        warnings.extend(item_warnings)

        # Cross-check line items sum vs subtotal / total
        if cleaned_items:
            items_sum = sum(
                item["total"] for item in cleaned_items if item.get("total") is not None
            )
            if subtotal_val is not None:
                if abs(items_sum - subtotal_val) > (len(cleaned_items) * 0.10):
                    warnings.append(
                        f"Line items sum ({items_sum:.2f}) deviates from extracted subtotal ({subtotal_val:.2f})."
                    )
            elif total_val is not None and tax_val is None:
                if abs(items_sum - total_val) > (len(cleaned_items) * 0.10):
                    warnings.append(
                        f"Line items sum ({items_sum:.2f}) deviates from extracted total ({total_val:.2f})."
                    )

        is_valid = len(errors) == 0 and len(warnings) <= 2 and total_val is not None

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "cleaned_fields": cleaned,
            "cleaned_line_items": cleaned_items,
        }

    def _validate_and_normalize_date(self, date_str: str) -> Tuple[Optional[str], Optional[str]]:
        """Parses arbitrary date formats into standard ISO YYYY-MM-DD."""
        try:
            # Clean ordinals like 1st, 2nd, 3rd
            clean_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str).strip()
            # If date starts with 4-digit year (YYYY-MM-DD or YYYY/MM/DD), force yearfirst=True, dayfirst=False
            if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", clean_str):
                dt = date_parser.parse(clean_str, fuzzy=True, yearfirst=True, dayfirst=False)
            else:
                dt = date_parser.parse(clean_str, fuzzy=True, dayfirst=True)
            if not (1990 <= dt.year <= 2040):
                return None, f"Year {dt.year} is out of expected invoice range."
            return dt.strftime("%Y-%m-%d"), None
        except Exception as e:
            return None, str(e)

    def _parse_numeric(self, val: Any) -> Optional[float]:
        """Safely parses currency/number string into float."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            # Strip currency symbols and whitespace
            s = str(val).strip()
            s = re.sub(r"[^\d\.,\-]", "", s)
            if not s:
                return None
            # Handle European 1.234,56 vs American 1,234.56
            if "," in s and "." in s:
                if s.rfind(",") > s.rfind("."):
                    # comma is decimal
                    s = s.replace(".", "").replace(",", ".")
                else:
                    s = s.replace(",", "")
            elif "," in s and "." not in s:
                # Comma might be decimal if 2 decimals at end
                parts = s.split(",")
                if len(parts) == 2 and len(parts[1]) == 2:
                    s = s.replace(",", ".")
                else:
                    s = s.replace(",", "")
            return round(float(s), 2)
        except Exception:
            return None

    def _validate_line_items(self, items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validates individual line items and per-item arithmetic."""
        cleaned_list: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for idx, itm in enumerate(items):
            desc = str(itm.get("description", "")).strip()
            qty = self._parse_numeric(itm.get("quantity")) or 1.0
            price = self._parse_numeric(itm.get("unit_price"))
            tax_rate = self._parse_numeric(itm.get("tax_rate"))
            tax_amt = self._parse_numeric(itm.get("tax_amount"))
            tot = self._parse_numeric(itm.get("total"))

            # Derive total if missing
            if tot is None and price is not None:
                tot = round(qty * price, 2)
            # Derive unit price if missing
            if price is None and tot is not None and qty > 0:
                price = round(tot / qty, 2)

            # Check math: qty * price == tot
            if price is not None and tot is not None:
                expected_tot = round(qty * price, 2)
                if abs(expected_tot - tot) > 0.05 and tax_amt is None:
                    warnings.append(
                        f"Line item {idx + 1} ('{desc[:20]}'): Qty ({qty}) * Price ({price:.2f}) "
                        f"= {expected_tot:.2f}, expected {tot:.2f}."
                    )

            cleaned_list.append({
                "description": desc,
                "quantity": qty,
                "unit_price": price,
                "tax_rate": tax_rate,
                "tax_amount": tax_amt,
                "total": tot,
            })

        return cleaned_list, warnings
