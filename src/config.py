"""
Configuration module for the ML-Based Invoice Data Extraction System.
Defines paths, regex patterns, field names, and system settings.
"""

from pathlib import Path
import os
import shutil

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
INVOICES_DIR = DATA_DIR / "invoices"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "invoice_crf_model.pkl"
OUTPUT_DIR = BASE_DIR / "output"
DOCS_DIR = BASE_DIR / "docs"

# Create essential directories
for directory in [DATA_DIR, INVOICES_DIR, ANNOTATIONS_DIR, SPLITS_DIR, MODELS_DIR, OUTPUT_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Supported input file formats
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
ARCHIVE_EXTENSIONS = {".zip"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS.union(ARCHIVE_EXTENSIONS)

# Key Target Fields
TARGET_FIELDS = [
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "buyer_name",
    "purchase_order_number",
    "currency",
    "subtotal",
    "tax_amount",
    "total_amount",
    "tax_id",  # GSTIN / Tax Identification Number
]

LINE_ITEM_FIELDS = [
    "description",
    "quantity",
    "unit_price",
    "tax_rate",
    "tax_amount",
    "total",
]

# CRF Label Mapping (BIO Scheme)
CRF_LABELS = [
    "O",
    "B-INV_NUM", "I-INV_NUM",
    "B-INV_DATE", "I-INV_DATE",
    "B-SUPPLIER", "I-SUPPLIER",
    "B-BUYER", "I-BUYER",
    "B-PO_NUM", "I-PO_NUM",
    "B-CURRENCY",
    "B-SUBTOTAL", "I-SUBTOTAL",
    "B-TAX_AMT", "I-TAX_AMT",
    "B-TOTAL", "I-TOTAL",
    "B-TAX_ID", "I-TAX_ID",
]

# Tesseract Executable Discovery
def get_tesseract_cmd() -> str:
    """Find tesseract executable on Windows or Linux."""
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Check in PATH
    path_bin = shutil.which("tesseract")
    if path_bin:
        return path_bin

    # Standard Windows install locations
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in windows_paths:
        if os.path.exists(p):
            return p
    return "tesseract"

TESSERACT_CMD = get_tesseract_cmd()

# Regex Patterns for Candidate Extraction and Validation
PATTERNS = {
    # Indian GSTIN format: 2 digits (state) + 5 alpha (PAN) + 4 digits (PAN) + 1 alpha (PAN) + 1 entity num + 'Z' + 1 checksum
    "gstin": r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b",
    
    # Generic Tax ID / VAT / EIN / Tax Reg format
    "tax_id": r"\b(?:TAX\s*(?:ID|NO|#)?|VAT\s*(?:NO|#)?|EIN|GSTIN|GST)\s*[:#\-]?\s*([A-Z0-9\-]{8,18})\b",
    
    # Invoice Number patterns
    "invoice_number": r"(?i)\b(?:invoice|inv|bill)\s*(?:no|number|num|#)?\s*[:#\-]?\s*([A-Za-z0-9\-_/]{3,25})\b",
    
    # Purchase Order patterns
    "po_number": r"(?i)\b(?:purchase\s*order|p\.?o\.?\s*(?:no|number|#)?)\s*[:#\-]?\s*([A-Za-z0-9\-_/]{3,20})\b",
    
    # Date formats (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, Month DD, YYYY, etc.)
    "date": r"\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{1,2}(?:st|nd|rd|th)?[\s,]+\d{4}|\d{1,2}(?:st|nd|rd|th)?[\s,]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4})\b",
    
    # Currency symbols & codes
    "currency_code": r"\b(USD|EUR|GBP|INR|CAD|AUD|SGD|JPY|CNY|CHF|NZD)\b",
    "currency_symbol": r"([\$€£₹¥])",
    
    # Numeric amounts (handles 1,234.56, 1234.56, 1.234,56 etc.)
    "amount": r"(?<!\w)(?:[\$€£₹¥]\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})|\d+(?:\.\d{1,2})?)(?!\w)",
}

# Currency Symbol to ISO Code Mapping
CURRENCY_MAP = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "¥": "JPY",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "INR": "INR",
    "CAD": "CAD",
    "AUD": "AUD",
    "SGD": "SGD",
}

# Numerical tolerance for tax and total balance checks (float rounding differences)
MATH_TOLERANCE = 0.05
