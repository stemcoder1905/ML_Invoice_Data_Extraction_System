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
    "taxable_amount",
    "total",
]

# Standard Field Aliases for Multi-Vendor Layout Normalization
FIELD_ALIASES = {
    "invoice_number": [
        "invoice no", "invoice number", "tax invoice no", "bill no", "document number",
        "invoice ref", "invoice #", "bill #", "receipt no", "inv no", "inv number",
        "tax invoice number", "bill number", "cash memo"
    ],
    "purchase_order_number": [
        "order no", "order id", "purchase order", "po number", "order reference",
        "po #", "order #", "booking id", "trip id", "p.o. no", "po no"
    ],
    "invoice_date": [
        "invoice date", "bill date", "date of issue", "tax invoice date", "dated",
        "date", "order date", "billing date"
    ],
    "supplier_name": [
        "seller name", "sold by", "vendor", "supplier", "issued by", "service provider",
        "merchant", "restaurant name", "store name", "seller"
    ],
    "buyer_name": [
        "bill to", "billed to", "customer name", "buyer name", "invoice to",
        "client name", "deliver to", "ship to", "customer", "buyer", "recipient"
    ],
    "subtotal": [
        "taxable value", "taxable amount", "assessable value", "taxable amt",
        "subtotal", "sub total", "sub-total", "net amount", "total taxable value"
    ],
    "tax_amount": [
        "total tax", "tax amount", "tax total", "total gst", "gst amount",
        "vat amount", "cgst + sgst", "igst amount"
    ],
    "total_amount": [
        "grand total", "invoice total", "total amount", "net payable", "amount payable",
        "total amt", "invoice value", "item total", "balance due", "total due"
    ],
    "tax_id": [
        "gstin", "gstin/uin", "supplier gstin", "seller gstin", "gst registration number",
        "tax id", "vat id", "ein", "tax reg no"
    ],
}

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
    "tax_id": r"\b(?:TAX\s*(?:ID|NO|#)?|VAT\s*(?:NO|#)?|EIN|GSTIN|GST)\s*[:#\-]?\s*([A-Z0-9\-]{7,18})\b",
    
    # Invoice Number patterns (labeled match preferred)
    "invoice_number": r"(?i)\b(?:tax\s*invoice|invoice|inv|bill|document|receipt)\s*(?:no\.?|number|num\.?|ref\.?|#)\s*[:#\-]?\s*([A-Za-z0-9\-_/]{3,35})\b",
    
    # Purchase Order & Order Number patterns
    "po_number": r"(?i)\b(?:purchase\s*order|p\.?o\.?\s*(?:no|number|#)?|order\s*(?:no\.?|id|number|ref\.?|#)|booking\s*id)\s*[:#\-]?\s*(PO-[A-Za-z0-9\-_/]+|[A-Za-z0-9\-_/]{4,35})\b",
    
    # Date formats (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, 24-Oct-2026, Month DD, YYYY, etc.)
    "date": r"\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,\-]+\d{1,2}(?:st|nd|rd|th)?[\s,\-]+\d{2,4}|\d{1,2}(?:st|nd|rd|th)?[\s,\-]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,\-]+\d{2,4})\b",
    
    # Currency symbols & codes
    "currency_code": r"\b(USD|EUR|GBP|INR|CAD|AUD|SGD|JPY|CNY|CHF|NZD|AED|SAR)\b",
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
    "RS": "INR",
    "RS.": "INR",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "INR": "INR",
    "CAD": "CAD",
    "AUD": "AUD",
    "SGD": "SGD",
    "AED": "AED",
    "SAR": "SAR",
}

# Numerical tolerance for tax and total balance checks (float rounding differences)
MATH_TOLERANCE = 0.05
