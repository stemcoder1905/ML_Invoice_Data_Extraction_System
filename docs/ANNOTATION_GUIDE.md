# Invoice Annotation Guide

## 1. Overview
This document specifies the ground-truth annotation standard for the **ML-Based Invoice Data Extraction System**. To train and evaluate the sequence-labeling Conditional Random Field (CRF) model without data leakage, annotations are stored per-document in standard JSON format under `data/annotations/{doc_id}.json`.

---

## 2. Annotation File Schema

Each annotation file contains three principal keys:
1. `document_id`: Unique identifier (e.g., `INV_0001`).
2. `ground_truth_fields`: Normalized target field values for document-level evaluation.
3. `ground_truth_line_items`: Array of tabular line item records.
4. `tokens`: Token-level sequence with normalized bounding boxes and BIO labels for CRF training.

### JSON Structure Example
```json
{
  "document_id": "INV_0001",
  "file_name": "INV_0001.pdf",
  "ground_truth_fields": {
    "invoice_number": "INV-2024-5821",
    "invoice_date": "2024-06-14",
    "supplier_name": "Vertex Global Solutions Ltd",
    "buyer_name": "Acme Retail Enterprises",
    "purchase_order_number": "PO-2024-412",
    "currency": "INR",
    "subtotal": 1420.00,
    "tax_amount": 255.60,
    "total_amount": 1675.60,
    "tax_id": "27AABCU9603R1ZM"
  },
  "ground_truth_line_items": [
    {
      "description": "Cloud Server Compute (vCPU/RAM)",
      "quantity": 2.0,
      "unit_price": 120.00,
      "tax_rate": 0.18,
      "tax_amount": 43.20,
      "total": 283.20
    }
  ],
  "tokens": [
    {
      "text": "Invoice",
      "clean_text": "Invoice",
      "page_num": 1,
      "bbox": [435.0, 75.0, 480.0, 88.0],
      "norm_bbox": [0.710, 0.094, 0.784, 0.111],
      "bio_tag": "O"
    },
    {
      "text": "INV-2024-5821",
      "clean_text": "INV-2024-5821",
      "page_num": 1,
      "bbox": [485.0, 75.0, 560.0, 88.0],
      "norm_bbox": [0.792, 0.094, 0.915, 0.111],
      "bio_tag": "B-INV_NUM"
    }
  ]
}
```

---

## 3. BIO Tagging Scheme

Tokens are labeled using the standard Beginning-Inside-Outside (**BIO**) scheme:

| Tag | Target Field | Description |
| :--- | :--- | :--- |
| `B-INV_NUM`, `I-INV_NUM` | `invoice_number` | Invoice serial or reference number |
| `B-INV_DATE`, `I-INV_DATE` | `invoice_date` | Document issue date |
| `B-SUPPLIER`, `I-SUPPLIER` | `supplier_name` | Vendor / seller registered business name |
| `B-BUYER`, `I-BUYER` | `buyer_name` | Customer / recipient registered name |
| `B-PO_NUM`, `I-PO_NUM` | `purchase_order_number` | Associated Purchase Order identifier |
| `B-CURRENCY` | `currency` | Currency symbol or ISO-4217 code (USD, INR, EUR) |
| `B-SUBTOTAL`, `I-SUBTOTAL` | `subtotal` | Pre-tax net invoice taxable value |
| `B-TAX_AMT`, `I-TAX_AMT` | `tax_amount` | Total aggregated tax (GST, VAT, Sales Tax) |
| `B-TOTAL`, `I-TOTAL` | `total_amount` | Final payable balance |
| `B-TAX_ID`, `I-TAX_ID` | `tax_id` | Vendor Tax ID, GSTIN, EIN, or VAT registration |
| `O` | *Non-target* | Background text, column headers, terms, noise |

---

## 4. Multi-Token Entity Annotation Rules

1. **Entity Boundaries**: The first token of an entity sequence **must** carry the `B-` prefix. All subsequent contiguous tokens belonging to the same entity take `I-`.
   - *Example:* `Vertex` (`B-SUPPLIER`), `Global` (`I-SUPPLIER`), `Solutions` (`I-SUPPLIER`), `Ltd` (`I-SUPPLIER`).
2. **Punctuation Trimming**: Punctuation delimiters (such as trailing colons or commas) attached to prompt keywords are tagged as `O`.
   - *Example:* In `Invoice No: INV-100`, `Invoice` is `O`, `No:` is `O`, and `INV-100` is `B-INV_NUM`.
3. **Compound Numeric Amounts**: Currency symbols attached directly to numeric strings are separated or tagged appropriately.
   - *Example:* In `$1,250.00`, if segmented together, it is tagged `B-TOTAL`; if split, `$` is `B-CURRENCY` and `1,250.00` is `B-TOTAL`.

---

## 5. Document-Level Splitting Rules (Strict Zero-Leakage)

- **NEVER** randomly partition token lists across splits.
- Splitting is performed exclusively at the **Document Level**:
  - `data/splits/train_docs.json`: Invoices allocated exclusively to model training.
  - `data/splits/test_docs.json`: Invoices reserved strictly for held-out evaluation.
- No invoice ID in the test set may appear in the training split.
