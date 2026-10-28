# Windows Setup and Tesseract OCR Installation Guide

This guide details the step-by-step instructions for configuring and executing the **ML-Based Invoice Data Extraction System** on Windows.

---

## 1. Prerequisites
- **Windows 10 or 11 (64-bit)**
- **Python 3.10+** (ensure "Add Python to PATH" was selected during install)
- PowerShell or Windows Terminal

---

## 2. Virtual Environment Setup

Open PowerShell in the project root directory (`D:\OCR_project`):

```powershell
# Navigate to the project folder
cd D:\OCR_project

# Create a clean virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install all project dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PowerShell Script Execution Policy**:  
> If you encounter `cannot be loaded because running scripts is disabled on this system`, run:  
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## 3. Installing Tesseract OCR on Windows

For digital PDFs (which account for standard modern e-invoices), **PyMuPDF extracts native text directly without requiring Tesseract**. However, for scanned PDFs and raster image invoices (PNG/JPG), Tesseract OCR is utilized.

### Method A: Via Windows Package Manager (`winget`) — Recommended
Run the following command in PowerShell:
```powershell
winget install --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
```

### Method B: Manual Installer
1. Download the latest Windows 64-bit installer from the official UB-Mannheim repository:  
   👉 [UB-Mannheim Tesseract Releases](https://github.com/UB-Mannheim/tesseract/releases)
2. Run `tesseract-ocr-w64-setup-v5.x.exe`.
3. Default installation path:  
   `C:\Program Files\Tesseract-OCR\tesseract.exe`
4. Add `C:\Program Files\Tesseract-OCR` to your Windows System `PATH` environment variable.

### Verifying Tesseract in PowerShell:
```powershell
tesseract --version
```

If Tesseract is installed in a custom directory, set the environment variable:
```powershell
$env:TESSERACT_CMD = "C:\Path\To\tesseract.exe"
```
The application automatically checks standard Windows installation paths.

---

## 4. Running the Complete System

### Step 1: Generate Dataset & Annotations
```powershell
python generate_dataset.py
```
*Generates 25 realistic demonstration invoices (PDF & PNG) with exact token bounding boxes and 70/30 train/test split.*

### Step 2: Train the CRF Sequence Model
```powershell
python train.py
```
*Trains the Conditional Random Field model on the training split documents and saves the artifact to `models/invoice_crf_model.pkl`.*

### Step 3: Run Evaluation on the Held-Out Test Set
```powershell
python evaluate.py
```
*Computes field-level Precision, Recall, F1-score, and Exact Match accuracy on held-out test invoices.*

### Step 4: Run Automated Tests
```powershell
pytest tests -v
```
*Executes all unit, functional, and integration tests.*

### Step 5: Launch the Streamlit User Interface
```powershell
streamlit run app.py
```
*Opens the web dashboard in your default browser at `http://localhost:8501`.*

---

## 5. Batch Processing from CLI
To process a folder of invoices from Python:
```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor()
summary = processor.process_batch("data/invoices")
print(f"Processed {summary['total_documents']} documents. All JSON outputs saved in output/")
```
