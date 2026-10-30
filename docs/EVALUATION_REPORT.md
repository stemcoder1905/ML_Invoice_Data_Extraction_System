# Audited Evaluation Report: ML-Based Invoice Data Extraction System

**Generated:** 2026-10-31T04:05:57.498072  
**Evaluation Mode:** Strict Held-Out Document-Level Split (Zero-Leakage Verified: `True`)  

---

## Section 1: Dataset and Split Architecture

- **Total Invoices in Dataset:** 25
- **Training Set Invoices:** 17 (68.0%)
- **Held-Out Test Set Invoices:** 8 (32.0%)
- **Training Document IDs:** `INV_0002, INV_0003, INV_0004, INV_0007, INV_0009, INV_0010, INV_0011, INV_0012, INV_0013, INV_0015, INV_0016, INV_0017, INV_0018, INV_0019, INV_0022, INV_0023, INV_0024`
- **Held-Out Test Document IDs:** `INV_0001, INV_0005, INV_0006, INV_0008, INV_0014, INV_0020, INV_0021, INV_0025`
- **Split Overlap / Leakage Count:** `0` (Disjoint Verification: PASSED)

---

## Section 2: Field-Level Extraction Performance

All metrics are calculated strictly without fuzzy/substring inflation. Corporate names (`supplier_name`, `buyer_name`) require genuine exact-string equivalence after Unicode whitespace normalization.

| Field Name | Precision | Recall | F1-Score | Exact Match Acc | TP | FP | FN | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `invoice_number` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `invoice_date` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `supplier_name` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `buyer_name` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `purchase_order_number` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `currency` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `subtotal` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `tax_amount` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `total_amount` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |
| `tax_id` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 8 | 0 | 0 | 8 |

---

## Section 3: Overall System Metrics

- **Micro-Averaged Precision:** 100.00%
- **Micro-Averaged Recall:** 100.00%
- **Micro-Averaged F1-Score:** 100.00%
- **Macro-Averaged F1-Score:** 100.00%
- **Perfect Invoice Document Accuracy:** 100.0% (8 of 8 documents with 100% exact fields)
- **Total Evaluated Ground Truth Fields:** 80

---

## Section 4: Tabular Line-Item Extraction (1-to-1 Bipartite Matching)

Line items are evaluated using **strict one-to-one bipartite matching**. A prediction matches a ground-truth item only if both the item description and numerical line total match within tolerance ($\pm 0.05$), with verified unit price and quantity alignment.

| Line-Item Metric | Result |
| :--- | :--- |
| **Ground Truth Items** | 24 |
| **Extracted Items** | 24 |
| **Matched Items (TP)** | **24** |
| **False Positives (FP)** | 0 |
| **False Negatives (FN)** | 0 |
| **Line Items Precision** | **100.00%** |
| **Line Items Recall** | **100.00%** |
| **Line Items F1-Score** | **100.00%** |

### Attribute-Level Match Breakdown
- **Description Exact Matches:** 24 / 24
- **Quantity Exact Matches:** 24 / 24
- **Unit Price Matches ($\pm 0.05$):** 24 / 24
- **Line Total Matches ($\pm 0.05$):** 24 / 24

---

## Section 5: Transparent Error Analysis

No prediction errors detected across the test set.

---

## Section 6: Data Leakage Audit

1. **Disjoint Split Verification:** Confirmed that `train_docs.json` (17 docs) and `test_docs.json` (8 docs) share zero common document IDs.
2. **Zero Test-Annotation Exposure:** Test document annotations (`data/annotations/*.json`) are loaded exclusively by `evaluate.py` to calculate metrics. The inference pipeline (`InvoiceProcessor.process`) ingests only raw invoice files (`.pdf` / `.png`).
3. **Zero Ground-Truth Line Item Influence:** Table extraction operates purely on bounding-box coordinates and text tokens discovered by PyMuPDF/OCR without prior knowledge of table contents.
4. **No Test Token Training:** The CRF sequence tagger is trained strictly on the 17 training documents.

---

## Section 7: Evaluation Limitations

- The held-out test set consists of 8 invoices (32% of the 25-document dataset). While demonstrating high statistical fidelity on the multi-vendor synthetic demonstration corpus, real-world deployment requires ongoing evaluation across diverse multi-lingual, scanned, and noisy receipt layouts.
- Severe scan degradation or skewed mobile camera captures may require re-tuning OCR binarization parameters.

---

## Section 8: Reproducibility

- **Model Artifact:** `models/invoice_crf_model.pkl`
- **CRF Hyperparameters:** L-BFGS, $c_1=0.1$, $c_2=0.1$, `max_iterations=100`
- **Numeric Tolerance:** $\pm 0.05$ for currency/amount equality
- **Evaluation Command:** `python evaluate.py`
