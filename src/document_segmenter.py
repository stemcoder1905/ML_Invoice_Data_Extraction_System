"""
Document Segmentation Module.
Detects document boundaries within a multi-page PDF to separate multiple financial
documents (e.g. Product Tax Invoice, Transport Bill of Supply, Platform Fee Invoice).
Uses page counters ('1 of 1', '1 of 2'), distinct invoice/bill numbers, document headers,
and seller changes to partition pages into coherent single-document units.
"""

from typing import List, Dict, Any, Tuple, Optional
import re


class DocumentSegmenter:
    """Segments a document's tokens and text across pages into distinct financial document records."""

    def __init__(self):
        self.re_page_counter = re.compile(
            r"(?i)(?:\bpage\s*(\d+)\s*(?:of|\/)\s*(\d+)\b|(?<!\w)(\d+)\s+(?:of|\/)\s+(\d+)(?!\w))"
        )
        self.re_doc_title = re.compile(
            r"(?i)\b(Tax\s+Invoice\s*\/\s*Bill\s+of\s+Supply|Tax\s+Invoice|Bill\s+of\s+Supply|Cash\s+Memo)\b"
        )
        self.re_inv_number = re.compile(
            r"(?i)\b(?:invoice\s*number\s*#|invoice\s*no\.?|invoice\s*number|bill\s*of\s*supply\s*number|tax\s*invoice\s*no\.?)\s*[:#\s-]*([A-Za-z0-9\-_/]{4,25})\b"
        )
        self.re_seller = re.compile(
            r"(?i)\b(?:sold\s*by|billed\s*from|seller\s*name|legal\s*entity\s*name)\s*[:\-]?\s*([A-Za-z0-9\s.,&'\-]{3,50})"
        )

    def segment_pages(
        self, tokens: List[Dict[str, Any]], page_count: int
    ) -> List[Dict[str, Any]]:
        """
        Analyzes tokens per page and groups consecutive pages into discrete document units.
        Returns a list of segment descriptors:
        [
            {
                "segment_index": 1,
                "page_numbers": [1],
                "document_type": "Tax Invoice",
                "invoice_number_hint": "LIADGPD270000018",
                "supplier_hint": "SPIGEN INDIA PRIVATE LIMITED",
            },
            ...
        ]
        """
        if page_count <= 1:
            return [{
                "segment_index": 1,
                "page_numbers": [1],
                "document_type": "Invoice",
                "invoice_number_hint": None,
                "supplier_hint": None,
            }]

        # Group tokens and text per page
        page_info: List[Dict[str, Any]] = []
        for p in range(1, page_count + 1):
            p_tokens = [t for t in tokens if t.get("page_num", 1) == p]
            p_text = " ".join(t["clean_text"] for t in p_tokens)

            # Detect page counter (e.g. 1 of 1, 1 of 2, 2 of 2)
            cur_p, tot_p = None, None
            m_pc = self.re_page_counter.search(p_text)
            if m_pc:
                groups = [g for g in m_pc.groups() if g is not None]
                if len(groups) == 2:
                    try:
                        cur_p, tot_p = int(groups[0]), int(groups[1])
                    except ValueError:
                        pass

            # Detect document title
            m_title = self.re_doc_title.search(p_text)
            doc_type = m_title.group(1).title() if m_title else "Invoice"

            # Detect invoice/bill of supply number
            m_inv = self.re_inv_number.search(p_text)
            inv_num = m_inv.group(1).strip() if m_inv else None

            # Detect seller
            m_seller = self.re_seller.search(p_text)
            seller = m_seller.group(1).strip() if m_seller else None

            page_info.append({
                "page_num": p,
                "cur_p": cur_p,
                "tot_p": tot_p,
                "doc_type": doc_type,
                "inv_num": inv_num,
                "seller": seller,
                "tokens": p_tokens,
                "text": p_text,
            })

        # Cluster pages into documents
        segments: List[List[int]] = []
        cur_segment: List[int] = [1]

        for p_idx in range(1, page_count):
            prev = page_info[p_idx - 1]
            curr = page_info[p_idx]

            is_new_doc = False

            # Condition 1: Explicit page numbering (e.g. current page is "1 of N")
            if curr["cur_p"] == 1 and curr["tot_p"] is not None:
                is_new_doc = True
            # Condition 2: Current page is "2 of 2" or "K of N" where K > 1 -> strictly continuation
            elif curr["cur_p"] is not None and curr["cur_p"] > 1:
                is_new_doc = False
            # Condition 3: Different non-empty invoice / bill number
            elif curr["inv_num"] and prev["inv_num"] and curr["inv_num"] != prev["inv_num"]:
                is_new_doc = True
            # Condition 4: Different seller name and new doc title
            elif curr["seller"] and prev["seller"] and curr["seller"] != prev["seller"] and curr["doc_type"]:
                is_new_doc = True
            # Condition 5: Previous page was explicitly the terminal page ("N of N", e.g. "1 of 1" or "2 of 2")
            elif prev["cur_p"] is not None and prev["tot_p"] is not None and prev["cur_p"] == prev["tot_p"]:
                # If current page has an invoice number or document header, start new document
                if curr["inv_num"] or curr["seller"] or curr["doc_type"] != "Invoice":
                    is_new_doc = True

            if is_new_doc:
                segments.append(cur_segment)
                cur_segment = [curr["page_num"]]
            else:
                cur_segment.append(curr["page_num"])

        if cur_segment:
            segments.append(cur_segment)

        # Build structured segment descriptors
        result: List[Dict[str, Any]] = []
        for idx, seg_pages in enumerate(segments, 1):
            first_p = page_info[seg_pages[0] - 1]
            result.append({
                "segment_index": idx,
                "page_numbers": seg_pages,
                "document_type": first_p["doc_type"],
                "invoice_number_hint": first_p["inv_num"],
                "supplier_hint": first_p["seller"],
            })

        return result
