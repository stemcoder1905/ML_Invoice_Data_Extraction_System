# ML-Based Invoice Data Extraction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Zero-LLM Guarantee](https://img.shields.io/badge/Architecture-Traditional%20ML%20Only-green.svg)](#strict-assessment-compliance-zero-generative-ai)

A production-oriented, traditional Machine Learning and Computer Vision system for extracting structured header fields and tabular line items from invoice documents (PDF, PNG, JPG, JPEG).

Developed for an **AI/ML Intern Assessment**, strictly engineered using classical Machine Learning, sequence modeling, and optical character recognition.

---

## 🛡️ Strict Assessment Compliance: Zero Generative AI

> [!IMPORTANT]
> **NO Large Language Models (LLMs)**, Generative AI models, Gemini APIs, OpenAI APIs, Claude APIs, Vision-Language Models (VLMs), or generative document services are utilized.
>
> All extraction is powered exclusively by:
> 1. **Conditional Random Fields (CRF)** sequence labeling on engineered lexical, orthographic, domain, spatial, and contextual token features.
> 2. **PyMuPDF & Tesseract OCR** with OpenCV image deskewing, noise filtering, and adaptive binarization.
> 3. **Spatial Line-Item Table Extraction** using coordinate clustering and mathematical reconciliation.
> 4. **Complementary Anchored Regular Expressions** for rule-based verification.
> 5. **Financial Consistency & GSTIN Validation**.

---

## 🚀 Key Capabilities

- **Multi-Format Ingestion**: Supports digital PDFs, scanned PDFs, PNG, JPG, and JPEG invoices.
- **Computer Vision Preprocessing**: OpenCV-based resolution normalization, grayscale conversion, Gaussian noise reduction, contour skew detection (`minAreaRect`), deskew rotation, and Otsu/adaptive binarization.
- **Target Fields Extracted**:
  - `invoice_number`
  - `invoice_date` (normalized to ISO `YYYY-MM-DD`)
  - `supplier_name` (vendor / seller)
  - `buyer_name` (customer / client)
  - `purchase_order_number`
  - `currency` (ISO-4217 code or symbol)
  - `subtotal` (taxable value)
  - `tax_amount` (GST, VAT, or Sales Tax)
  - `total_amount` (gross payable)
  - `tax_id` (15-character Indian GSTIN or Tax Reg)
- **Tabular Line-Item Extraction**:
  - `description`
  - `quantity`
  - `unit_price`
  - `tax_rate` / `tax_amount`
  - `total`
- **Quality & Financial Validation**:
  - Reconciles $\text{Subtotal} + \text{Tax} = \text{Total}$ (flags mathematical discrepancies).
  - Validates GSTIN structure and check digits.
  - Line-item arithmetic check ($\text{Qty} \times \text{Price} = \text{Line Total}$).
- **Document-Level Train/Test Split (Zero-Leakage)**: Training and test sets are split strictly at the document level. Never leaks tokens between splits.
- **Interactive Streamlit Web UI**:
  - Single and batch invoice upload.
  - Visual preview of invoice documents.
  - Interactive, editable form for human-in-the-loop review.
  - Single and batch ZIP JSON download.
  - Real-time validation warning indicators.
  - Integrated evaluation dashboard.

---

## 🏛️ Pipeline Architecture

```
                                +---------------------------+
                                |      INVOICE DOCUMENT     |
                                | (Digital/Scanned PDF/Img) |
                                +---------------------------+
                                              |
                                              v
                                +---------------------------+
                                |    OPENCV PREPROCESSING   |
                                |  Deskew, Denoise, Otsu    |
                                +---------------------------+
                                              |
                                              v
                                +---------------------------+
                                |    OCR & WORD BOXES       |
                                |  PyMuPDF / Tesseract      |
                                +---------------------------+
                                              |
                                              v
                                +---------------------------+
                                |    FEATURE ENGINEERING    |
                                |  Lexical, Shape, Spatial, |
                                |  Contextual Windows       |
                                +---------------------------+
                                  /                       \
                                 v                         v
                   +------------------------+   +--------------------+
                   |     TRADITIONAL ML:    |   |     LINE ITEM      |
                   |   CRF SEQUENCE TAGGER  |   |  TABLE EXTRACTOR   |
                   | (sklearn-crfsuite BIO) |   |  (Spatial Cluster) |
                   +------------------------+   +--------------------+
                                 \                         /
                                  v                       v
                                +---------------------------+
                                |    HYBRID REFINEMENT      |
                                |   CRF + Anchored Regex    |
                                +---------------------------+
                                              |
                                              v
                                +---------------------------+
                                |   VALIDATION & QA ENGINE  |
                                |   GSTIN, Dates, Math Bal  |
                                +---------------------------+
                                              |
                                              v
                                +---------------------------+
                                |     RFC-COMPLIANT JSON    |
                                |    Extracted Output       |
                                +---------------------------+
```

---

## 📂 Project Directory Structure

```text
d:\OCR_project\
├── README.md                   # Project overview, architecture & instructions
├── requirements.txt            # Python dependencies
├── setup_windows.md            # Windows setup & Tesseract installation guide
├── pytest.ini                  # Pytest configuration
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # CRF model training script
├── evaluate.py                 # Evaluation script (held-out test split)
├── generate_dataset.py         # Synthetic invoice & annotation generator
├── src/
│   ├── __init__.py
│   ├── config.py               # Constants, paths, regex patterns, field names
│   ├── document_loader.py      # Ingests PDF, PNG, JPG, JPEG; digital vs scanned
│   ├── image_preprocessing.py  # OpenCV deskew, grayscale, threshold, noise filter
│   ├── ocr_engine.py           # PyMuPDF direct text/box extraction + Tesseract OCR
│   ├── feature_engineering.py  # Lexical, orthographic, spatial, domain CRF features
│   ├── ml_extractor.py         # CRF sequence labeling model (fit, predict, chunk)
│   ├── line_item_extractor.py  # Coordinate-based table row & column clustering
│   ├── candidate_extractor.py  # Deterministic regex & contextual lookarounds
│   ├── postprocessor.py        # Ensemble merging, fallback disclosure & metadata
│   ├── validator.py            # Date, amount, GSTIN validation & math consistency
│   ├── exporter.py             # RFC-compliant JSON serialization & sanitization
│   └── batch_processor.py      # Directory batch runner with fault tolerance
├── data/
│   ├── invoices/               # Sample invoice documents (.pdf, .png)
│   ├── annotations/            # JSON ground-truth token BIO annotations
│   └── splits/                 # train_docs.json & test_docs.json
├── models/
│   └── invoice_crf_model.pkl   # Serialized trained CRF model artifact
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py
│   ├── test_preprocessing.py
│   ├── test_ocr_engine.py
│   ├── test_feature_engineering.py
│   ├── test_ml_extractor.py
│   ├── test_line_item_extractor.py
│   ├── test_validator.py
│   ├── test_exporter.py
│   └── test_batch_processor.py
├── docs/
│   ├── ANNOTATION_GUIDE.md     # Ground-truth schema & BIO tagging standard
│   ├── MODEL_DOCUMENTATION.md  # CRF architecture & feature engineering guide
│   ├── EVALUATION_REPORT.md    # Test split performance metrics report
│   └── KNOWN_LIMITATIONS.md    # System boundaries & edge-case analysis
└── output/
    └── evaluation_report.json  # Raw evaluation metrics JSON
```

---

## ⚡ Quickstart Instructions

### 1. Installation
Clone or navigate to the repository and install requirements:
```bash
pip install -r requirements.txt
```
*(On Windows, see [setup_windows.md](setup_windows.md) for virtual environment and Tesseract setup).*

### 2. Generate Demonstration Dataset
```bash
python generate_dataset.py
```
Outputs 25 invoices with full annotations split into `train_docs.json` (17 docs) and `test_docs.json` (8 docs).

### 3. Train the Traditional ML Model (CRF)
```bash
python train.py
```
Trains the sequence labeling model and serializes it to `models/invoice_crf_model.pkl`.

### 4. Evaluate on Held-Out Test Set
```bash
python evaluate.py
```
Computes field-level Precision, Recall, F1-Score, and Exact Match on unseen test invoices.

### 5. Run Automated Tests
```bash
pytest tests -v
```
Executes all 23 unit and integration tests.

### 6. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Opens the interactive web UI at `http://localhost:8501`.

---

## 📊 Evaluation Results Summary

Tested on **8 held-out invoices** (80 target field instances, 24 line items):

| Target Field | Precision | Recall | F1-Score | Exact Match Acc |
| :--- | :---: | :---: | :---: | :---: |
| `invoice_number` | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `invoice_date` | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `supplier_name` | **87.5%** | **87.5%** | **87.5%** | **87.5%** |
| `buyer_name` | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `purchase_order_number` | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `currency` | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `subtotal` | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `tax_amount` | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `total_amount` | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `tax_id` (GSTIN) | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **OVERALL (Micro-Average)** | **98.8%** | **98.8%** | **98.8%** | — |
| **OVERALL (Macro-Average F1)** | — | — | **98.8%** | — |
| **PERFECT DOCUMENT ACCURACY** | — | — | — | **87.5%** |

- **Line-Item Table Metrics**: 24/24 line items matched using strict 1-to-1 bipartite matching (**100.0% Precision, 100.0% Recall, 100.0% F1**).
- Full audited detailed breakdown and error analysis in [reports/evaluation_report.md](reports/evaluation_report.md) and [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md).

---

## 🧾 Sample Extracted JSON Output

```json
{
  "metadata": {
    "filename": "INV_0001.pdf",
    "document_type": "digital_pdf",
    "page_count": 1,
    "token_count": 82,
    "ocr_engine": "native_pdf",
    "extraction_method": "crf_ml",
    "processing_time_ms": 28.5,
    "timestamp": "2026-10-28T09:22:00Z"
  },
  "fields": {
    "invoice_number": "INV-2024-5821",
    "invoice_date": "2024-06-14",
    "supplier_name": "Vertex Global Solutions Ltd",
    "buyer_name": "Acme Retail Enterprises",
    "purchase_order_number": "PO-2024-412",
    "currency": "INR",
    "subtotal": 1420.0,
    "tax_amount": 255.6,
    "total_amount": 1675.6,
    "tax_id": "27AABCU9603R1ZM",
    "tax_id_type": "GSTIN"
  },
  "line_items": [
    {
      "description": "Cloud Server Compute (vCPU/RAM)",
      "quantity": 2.0,
      "unit_price": 120.0,
      "tax_rate": 0.18,
      "tax_amount": 43.2,
      "total": 283.2
    }
  ],
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": []
  }
}
```

---

## 📦 Batch Processing Example

```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor()
summary = processor.process_batch("data/invoices")

print(f"Total processed: {summary['total_documents']}")
print(f"Valid invoices:  {summary['valid_invoices']}")
print(f"Average time:    {summary['average_time_per_document_ms']} ms/doc")
```

---

## 📄 License
MIT License. Open for educational and intern assessment evaluation.
