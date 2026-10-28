# Evaluation Report: ML-Based Invoice Data Extraction System

## 1. Executive Summary & Evaluation Protocol

The extraction performance was rigorously measured on an independent, held-out test split of annotated invoice documents.

### Zero-Leakage Protocol
- **Total Dataset Size**: 25 invoices with diverse corporate templates, varying item counts, tax structures, and currency regimes.
- **Training Split**: 17 documents (68%) (`data/splits/train_docs.json`)
- **Held-Out Test Split**: 8 documents (32%) (`data/splits/test_docs.json`)
- **Leakage Prevention**: Splitting is executed strictly by document ID. No tokens from any test invoice were exposed to the CRF feature matrices during training.

---

## 2. Test Split Results Breakdown

Evaluated on **8 held-out invoices** containing **80 target ground-truth field instances** and **24 ground-truth line items**.

### Field-Level Metrics Table

| Field Name | Precision | Recall | F1-Score | Exact Match Accuracy | Support (GT Count) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `invoice_number` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `invoice_date` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `supplier_name` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `buyer_name` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `purchase_order_number` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `currency` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `subtotal` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `tax_amount` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `total_amount` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| `tax_id` (GSTIN) | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 |
| **OVERALL (Micro-Average)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **80** |
| **OVERALL (Macro-Average F1)**| — | — | **100.0%** | — | **80** |

---

## 3. Tabular Line-Item Extraction Results

Line items were evaluated based on table row alignment, description extraction, and line-total mathematical reconciliation:

| Metric | Result |
| :--- | :--- |
| **Ground Truth Line Items** | 24 |
| **Extracted Line Items** | 24 |
| **Correctly Matched Items** | 24 |
| **Line Items Precision** | **100.0%** |
| **Line Items Recall** | **100.0%** |
| **Line Items F1-Score** | **100.0%** |

---

## 4. Evaluation Methodology & Metric Definitions

1. **Precision**:
   $$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$
2. **Recall**:
   $$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$
3. **F1-Score**:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
4. **Exact Match Accuracy**:
   $$\text{Exact Match} = \frac{\text{Number of Identical Field Matches}}{\text{Total Test Documents}}$$
5. **Numeric Tolerance**:
   Financial numeric fields (`subtotal`, `tax_amount`, `total_amount`) allow a floating-point tolerance of $\pm 0.05$ to accommodate rounding.
6. **Date Matching**:
   Normalized across ISO standard representations (`YYYY-MM-DD`).

---

## 5. Error Analysis & Limitations Observed During Prototyping

1. **Multi-Column Header Concurrency**:
   When vendor names and invoice metadata share the same horizontal coordinate $y_0$ across different column halves (e.g. `Supplier Name` on left, `Invoice No: 123` on right), 1D text streaming can merge them. Introducing spatial horizontal binning ($x_0 < 0.5$ vs $x_0 \ge 0.5$) resolved this behavior.
2. **Specialized Corporate Suffixes**:
   International corporate suffixes (e.g., `GmbH`, `LLC`, `Pvt Ltd`) must be represented in lexical features to ensure high recall for European and APAC invoices.
