"""
Synthetic Invoice and Annotation Generator.
Generates realistic multi-template invoice documents (PDF and image formats),
computes ground-truth tokens with BIO tags, creates document-level train/test splits,
and provides a clean demonstration dataset for training and evaluation.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pymupdf as fitz  # PyMuPDF

from src.config import INVOICES_DIR, ANNOTATIONS_DIR, SPLITS_DIR, CURRENCY_MAP


# Sample realistic vendor entities
SUPPLIERS = [
    {"name": "Vertex Global Solutions Ltd", "gstin": "27AABCU9603R1ZM", "city": "Mumbai", "currency": "INR", "tax_label": "GSTIN"},
    {"name": "Nexus Cloud Technologies Inc", "gstin": "07AAACN1234F1Z5", "city": "New Delhi", "currency": "USD", "tax_label": "Tax ID"},
    {"name": "Apex Industrial Supplies Corp", "gstin": "29AABCA5678E1Z8", "city": "Bengaluru", "currency": "INR", "tax_label": "GSTIN"},
    {"name": "Horizon Digital Media LLC", "gstin": "33AAACH9999K1Z2", "city": "Chennai", "currency": "USD", "tax_label": "EIN"},
    {"name": "Pioneer Logistics & Freight Services", "gstin": "06AAACP4444L1Z1", "city": "Gurugram", "currency": "INR", "tax_label": "GSTIN"},
    {"name": "Zenith Software Systems GmbH", "gstin": "24AAACZ3333M1Z9", "city": "Ahmedabad", "currency": "EUR", "tax_label": "VAT ID"},
    {"name": "Quantum Precision Hardware Pvt Ltd", "gstin": "19AABCQ7777N1Z4", "city": "Kolkata", "currency": "INR", "tax_label": "GSTIN"},
    {"name": "Sterling Consulting Group Co", "gstin": "36AABCS1111P1Z6", "city": "Hyderabad", "currency": "GBP", "tax_label": "Tax Reg"},
]

# Sample realistic buyer entities
BUYERS = [
    {"name": "Acme Retail Enterprises", "city": "Pune", "po_prefix": "PO-2024-"},
    {"name": "BlueStar Enterprise Solutions", "city": "Noida", "po_prefix": "ORD-88-"},
    {"name": "OmniCorp Infrastructure Ltd", "city": "Mumbai", "po_prefix": "PO-99-"},
    {"name": "Velocity FinTech Labs", "city": "Bengaluru", "po_prefix": "VFL-PO-"},
    {"name": "Global Trade Logistics Inc", "city": "Delhi", "po_prefix": "GTL-"},
    {"name": "Frontier BioScience Research", "city": "Hyderabad", "po_prefix": "BIO-PO-"},
]

# Sample product/service line items catalog
CATALOG = [
    ("Cloud Server Compute (vCPU/RAM)", 120.00, 0.18),
    ("Managed Database Hosting", 250.00, 0.18),
    ("Enterprise Security License", 450.00, 0.18),
    ("Technical Support & Maintenance", 300.00, 0.18),
    ("Software Architecture Consulting", 500.00, 0.18),
    ("Network Firewall Appliance", 850.00, 0.18),
    ("Data Backup & Recovery Storage", 90.00, 0.18),
    ("API Integration Gateway Service", 180.00, 0.18),
]


def create_invoice_pdf(
    pdf_path: Path,
    inv_data: Dict[str, Any],
) -> None:
    """Draws a professional, clean invoice layout on PDF canvas."""
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Header / Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "TAX INVOICE")

    # Supplier Details (Top Left)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 80, inv_data["supplier_name"])
    c.setFont("Helvetica", 9)
    c.drawString(50, height - 95, f"Industrial Area, Phase II, {inv_data['city']}")
    c.drawString(50, height - 110, f"{inv_data['tax_label']}: {inv_data['tax_id']}")

    # Invoice Metadata (Top Right)
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(width - 50, height - 80, f"Invoice No: {inv_data['invoice_number']}")
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 50, height - 95, f"Invoice Date: {inv_data['invoice_date']}")
    c.drawRightString(width - 50, height - 110, f"PO Number: {inv_data['purchase_order_number']}")

    # Divider line
    c.setLineWidth(1)
    c.line(50, height - 125, width - 50, height - 125)

    # Buyer / Bill To Section
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height - 145, "Bill To:")
    c.setFont("Helvetica", 9)
    c.drawString(50, height - 160, inv_data["buyer_name"])
    c.drawString(50, height - 175, f"Tech Park, {inv_data['buyer_city']}")

    # Table Column Headers
    table_top = height - 210
    c.setFillColorRGB(0.92, 0.94, 0.96)
    c.rect(50, table_top - 5, width - 100, 20, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 9)
    c.drawString(55, table_top, "Description")
    c.drawRightString(340, table_top, "Quantity")
    c.drawRightString(420, table_top, "Unit Price")
    c.drawRightString(480, table_top, "Tax")
    c.drawRightString(width - 55, table_top, "Total")

    # Line Items
    y = table_top - 20
    c.setFont("Helvetica", 9)
    for itm in inv_data["line_items"]:
        c.drawString(55, y, itm["description"])
        c.drawRightString(340, y, str(int(itm["quantity"])))
        c.drawRightString(420, y, f"{itm['unit_price']:.2f}")
        c.drawRightString(480, y, f"{itm['tax_amount']:.2f}")
        c.drawRightString(width - 55, y, f"{itm['total']:.2f}")
        y -= 20

    # Summary section (Bottom Right)
    summary_top = y - 10
    c.setLineWidth(0.5)
    c.line(320, summary_top + 10, width - 50, summary_top + 10)

    c.setFont("Helvetica", 9)
    c.drawString(340, summary_top, "Subtotal:")
    c.drawRightString(width - 55, summary_top, f"{inv_data['subtotal']:.2f}")

    c.drawString(340, summary_top - 18, "Tax Amount:")
    c.drawRightString(width - 55, summary_top - 18, f"{inv_data['tax_amount']:.2f}")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(340, summary_top - 38, f"Total Amount ({inv_data['currency']}):")
    c.drawRightString(width - 55, summary_top - 38, f"{inv_data['total_amount']:.2f}")

    # Footer note
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 40, "Thank you for your business. Terms: Net 30 days.")

    c.save()


def label_tokens_bio(words: List[Any], inv_data: Dict[str, Any], page_w: float, page_h: float) -> List[Dict[str, Any]]:
    """
    Extracts native tokens and assigns exact ground-truth BIO tags based on ground truth fields.
    """
    labeled_tokens: List[Dict[str, Any]] = []

    # Map field values into search tokens
    field_matchers = [
        ("INV_NUM", inv_data["invoice_number"].split()),
        ("INV_DATE", inv_data["invoice_date"].split()),
        ("SUPPLIER", inv_data["supplier_name"].split()),
        ("BUYER", inv_data["buyer_name"].split()),
        ("PO_NUM", inv_data["purchase_order_number"].split()),
        ("CURRENCY", [inv_data["currency"]]),
        ("SUBTOTAL", [f"{inv_data['subtotal']:.2f}"]),
        ("TAX_AMT", [f"{inv_data['tax_amount']:.2f}"]),
        ("TOTAL", [f"{inv_data['total_amount']:.2f}"]),
        ("TAX_ID", [inv_data["tax_id"]]),
    ]

    for w_tuple in words:
        x0, y0, x1, y1, text, block_num, line_num, word_num = w_tuple[:8]
        clean_text = text.strip()
        if not clean_text:
            continue

        norm_bbox = [
            max(0.0, min(1.0, x0 / page_w)),
            max(0.0, min(1.0, y0 / page_h)),
            max(0.0, min(1.0, x1 / page_w)),
            max(0.0, min(1.0, y1 / page_h)),
        ]

        tok = {
            "text": text,
            "clean_text": clean_text,
            "page_num": 1,
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "norm_bbox": norm_bbox,
            "bio_tag": "O",
        }
        labeled_tokens.append(tok)

    # Assign BIO tags by matching sequences in reading order
    for tag_name, target_words in field_matchers:
        if not target_words:
            continue
        m_len = len(target_words)
        for i in range(len(labeled_tokens) - m_len + 1):
            window_texts = [
                labeled_tokens[i + k]["clean_text"].strip(":,;()") for k in range(m_len)
            ]
            target_clean = [tw.strip(":,;()") for tw in target_words]

            if [w.lower() for w in window_texts] == [tw.lower() for tw in target_clean]:
                # Found exact sequence match
                labeled_tokens[i]["bio_tag"] = f"B-{tag_name}"
                for k in range(1, m_len):
                    labeled_tokens[i + k]["bio_tag"] = f"I-{tag_name}"
                break

    return labeled_tokens


def generate_synthetic_dataset(num_invoices: int = 25, seed: int = 42) -> Dict[str, Any]:
    """Generates synthetic invoices, renders PDFs and images, and creates train/test split."""
    random.seed(seed)
    doc_ids = []

    print(f"Generating {num_invoices} demonstration invoices...")

    for i in range(1, num_invoices + 1):
        doc_id = f"INV_{i:04d}"
        doc_ids.append(doc_id)

        sup = random.choice(SUPPLIERS)
        buyer = random.choice(BUYERS)

        inv_num = f"INV-{2024}-{random.randint(1001, 9999)}"
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        inv_date = f"{day:02d}/{month:02d}/2024"
        po_num = f"{buyer['po_prefix']}{random.randint(100, 999)}"

        # Select 2 to 4 line items
        num_items = random.randint(2, 4)
        chosen_items = random.sample(CATALOG, num_items)

        line_items = []
        subtotal = 0.0
        tax_total = 0.0

        for desc, base_price, tax_rate in chosen_items:
            qty = float(random.randint(1, 5))
            u_price = round(base_price * random.uniform(0.9, 1.2), 2)
            item_tax = round(qty * u_price * tax_rate, 2)
            item_total = round(qty * u_price + item_tax, 2)

            subtotal += qty * u_price
            tax_total += item_tax

            line_items.append({
                "description": desc,
                "quantity": qty,
                "unit_price": u_price,
                "tax_rate": tax_rate,
                "tax_amount": item_tax,
                "total": item_total,
            })

        subtotal = round(subtotal, 2)
        tax_total = round(tax_total, 2)
        total_amount = round(subtotal + tax_total, 2)

        inv_data = {
            "invoice_number": inv_num,
            "invoice_date": inv_date,
            "supplier_name": sup["name"],
            "city": sup["city"],
            "buyer_name": buyer["name"],
            "buyer_city": buyer["city"],
            "purchase_order_number": po_num,
            "currency": sup["currency"],
            "tax_label": sup["tax_label"],
            "tax_id": sup["gstin"],
            "subtotal": subtotal,
            "tax_amount": tax_total,
            "total_amount": total_amount,
            "line_items": line_items,
        }

        # 1. Render PDF
        pdf_path = INVOICES_DIR / f"{doc_id}.pdf"
        create_invoice_pdf(pdf_path, inv_data)

        # 2. Extract Native Tokens and Render Image
        doc = fitz.open(pdf_path)
        page = doc[0]
        words = page.get_text("words")
        labeled_tokens = label_tokens_bio(words, inv_data, page.rect.width, page.rect.height)

        # Convert 20% of generated documents to standalone images (PNG) to cover image inputs
        if i % 5 == 0:
            pix = page.get_pixmap(dpi=300)
            img_path = INVOICES_DIR / f"{doc_id}.png"
            pix.save(str(img_path))

        # 3. Save ground-truth annotation file
        annotation_payload = {
            "document_id": doc_id,
            "file_name": f"{doc_id}.pdf",
            "ground_truth_fields": {
                "invoice_number": inv_num,
                "invoice_date": "2024-" + f"{month:02d}-{day:02d}",
                "supplier_name": sup["name"],
                "buyer_name": buyer["name"],
                "purchase_order_number": po_num,
                "currency": sup["currency"],
                "subtotal": subtotal,
                "tax_amount": tax_total,
                "total_amount": total_amount,
                "tax_id": sup["gstin"],
            },
            "ground_truth_line_items": line_items,
            "tokens": labeled_tokens,
        }

        annot_path = ANNOTATIONS_DIR / f"{doc_id}.json"
        with open(annot_path, "w", encoding="utf-8") as f:
            json.dump(annotation_payload, f, indent=2, ensure_ascii=False)

    # Document-level Split: 70% Train, 30% Held-out Test
    random.shuffle(doc_ids)
    split_idx = int(0.70 * len(doc_ids))
    train_docs = sorted(doc_ids[:split_idx])
    test_docs = sorted(doc_ids[split_idx:])

    with open(SPLITS_DIR / "train_docs.json", "w", encoding="utf-8") as f:
        json.dump(train_docs, f, indent=2)

    with open(SPLITS_DIR / "test_docs.json", "w", encoding="utf-8") as f:
        json.dump(test_docs, f, indent=2)

    print(f"Generated {len(doc_ids)} documents.")
    print(f"Train split: {len(train_docs)} invoices saved to {SPLITS_DIR / 'train_docs.json'}")
    print(f"Held-out test split: {len(test_docs)} invoices saved to {SPLITS_DIR / 'test_docs.json'}")

    return {
        "total_documents": len(doc_ids),
        "train_count": len(train_docs),
        "test_count": len(test_docs),
    }


if __name__ == "__main__":
    generate_synthetic_dataset(num_invoices=25)
