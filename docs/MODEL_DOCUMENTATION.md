# Machine Learning Model Documentation

## 1. Traditional ML Architecture: Conditional Random Fields (CRF)

The extraction of invoice entities is modeled as a structured sequence labeling task using **Linear-Chain Conditional Random Fields (CRF)**, implemented via `sklearn-crfsuite` (wrapping `python-crfsuite`).

### Why Conditional Random Fields for Invoices?
Unlike traditional token-level classifiers (e.g., Logistic Regression or Naive Bayes), a CRF models dependencies between consecutive labels $P(\mathbf{y} \mid \mathbf{x})$. This prevents invalid BIO transition sequences (such as an `I-SUPPLIER` appearing immediately after an `O` without an antecedent `B-SUPPLIER`) and naturally captures spatial document reading order.

---

## 2. Feature Engineering Specification

For each document token $t_i$ at index $i$, an extensive set of engineered features is computed:

### A. Lexical & Orthographic Features
- `word.lower`: Lowercased string identity.
- `word.len`: Token length in characters.
- `word.isupper`, `word.islower`, `word.istitle`: Case capitalization flags.
- `word.isdigit`, `word.isalnum`: Numeric flags.
- `word.has_hyphen`, `word.has_slash`, `word.has_colon`: Structural separator flags.
- `word.shape`: Generalized character shape representation:
  - Uppercase $\to$ `X`, Lowercase $\to$ `x`, Digit $\to$ `9`, Punctuation preserved.
  - Runs of 4+ identical characters collapsed (e.g., `INV-2024-01` $\to$ `XXX-9+-99`).
- `prefix_2`, `prefix_3`, `suffix_2`, `suffix_3`: Sub-word n-gram affixes.

### B. Domain Keyword Lexicons
- `kw.invoice`: Matches invoice identifiers (`invoice`, `inv`, `bill`, `receipt`).
- `kw.date`: Matches date prompts (`date`, `dated`, `issue`, `due`).
- `kw.po`: Matches purchase order indicators (`po`, `order`, `p.o.`).
- `kw.total`: Matches balance keywords (`total`, `grand`, `payable`, `balance`).
- `kw.subtotal`: Matches net value keywords (`subtotal`, `taxable`, `net`).
- `kw.tax`: Matches tax designations (`tax`, `vat`, `gst`, `cgst`, `sgst`).
- `kw.company_sfx`: Corporate suffixes (`ltd`, `pvt`, `inc`, `corp`, `llc`, `gmbh`, `solutions`).
- `kw.currency`: Matches ISO codes (USD, EUR, INR) and symbols ($, €, ₹, £).

### C. Positional & Layout Features
- `pos.page_num`: Document page index.
- `pos.is_page_1`: Boolean indicator for first page.
- `pos.x0_bin`, `pos.y0_bin`: Quantized spatial grid coordinates (5% increments) normalized to $[0.0, 1.0]$.
- `pos.is_top_quarter`: $y_0 < 0.25$ (typical header region).
- `pos.is_top_half`: $y_0 < 0.50$.
- `pos.is_bottom_quarter`: $y_0 > 0.75$ (typical footer and summary total region).
- `pos.is_left_half`, `pos.is_right_half`: Horizontal sectioning.

### D. Contextual Window Features (Sliding Horizon $[-2, +2]$)
- Previous token $t_{i-1}$: Lowercase text, title case, digit status, colon presence, keyword matches, and horizontal displacement $\Delta x$.
- Pre-antecedent token $t_{i-2}$: Lowercase text and keyword indicators.
- Next token $t_{i+1}$: Lowercase text, title case, corporate suffixes, same-line indicator, and horizontal gap.
- Subsequent token $t_{i+2}$: Corporate suffixes and lexical identity.
- Boundary flags: `BOS` (Beginning of Sequence) and `EOS` (End of Sequence).

---

## 3. Training & Optimization Hyperparameters

- **Optimization Algorithm**: Limited-memory Broyden–Fletcher–Goldfarb–Shanno (**L-BFGS**)
- **L1 Regularization Penalty ($c_1$)**: `0.1` (induces sparsity over redundant features)
- **L2 Regularization Penalty ($c_2$)**: `0.1` (prevents feature weight divergence)
- **Maximum Iterations**: `100`
- **Transition States**: `all_possible_transitions=True` (learns valid BIO sequencing constraints)

---

## 4. Complementary Hybrid Candidate Extractor & Fallback

In adherence to production architectural standards:
1. **Primary**: The CRF sequence model tags entity spans.
2. **Complementary Candidate Matcher**: High-precision anchored regular expressions recover fields in cases of OCR token fragmentation or rare fonts.
3. **Disclosed Fallback**: When the CRF model is untrained or absent, the pipeline falls back gracefully to rule-based candidates and explicitly flags:
   `"CRF model is not trained or loaded. Extraction completed using rule-based candidate fallback."`

---

## 5. Third-Party Dependencies and Pretrained Components

| Component | Library / Source | Role | Pretrained Weights / License |
| :--- | :--- | :--- | :--- |
| **PyMuPDF** | `pymupdf` (MuPDF engine) | PDF document rendering & native text coordinate extraction | AGPL / Commercial |
| **Tesseract OCR** | `pytesseract` / Google Tesseract | Optical character recognition on raster images | Apache 2.0 (Pre-trained LSTM OCR engine) |
| **OpenCV** | `opencv-python` | Image preprocessing (deskew, binarization, filtering) | Apache 2.0 |
| **CRF Suite** | `sklearn-crfsuite` / `python-crfsuite` | Linear-chain CRF sequence tagging | MIT / BSD |
| **Scikit-Learn** | `scikit-learn` | Model utilities, evaluation matrices | BSD |
| **Date Parser** | `python-dateutil` | Fuzzy date parsing and normalization | Apache 2.0 / BSD |

> **STRICT COMPLIANCE DECLARATION**:  
> No Large Language Models (LLMs), Gemini APIs, OpenAI APIs, Claude APIs, vision-language models (VLMs), or generative document foundation models are used anywhere in this project.
