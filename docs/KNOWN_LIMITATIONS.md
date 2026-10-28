# Known Limitations & Edge Cases

While the **ML-Based Invoice Data Extraction System** provides robust performance using traditional ML and OCR techniques, certain boundary conditions and architectural trade-offs exist:

---

## 1. Physical Document Degradation
- **Severely Creased or Crumpled Paper**:
  Tesseract OCR accuracy degrades when scanned documents have severe non-linear folding shadows or watermarks that OpenCV adaptive thresholding cannot completely eliminate.
- **Low Resolution (<150 DPI)**:
  Images scanned below 150 DPI often exhibit broken character strokes, leading to OCR errors such as interpreting `8` as `B` or `0` as `O`.

---

## 2. Unconventional & Complex Table Topologies
- **Nested / Borderless Multi-Line Item Descriptions**:
  When line-item descriptions span multiple wrapped lines without horizontal cell borders, horizontal coordinate clustering may split a single description across multiple candidate rows.
- **Rotated or Diagonal Invoices**:
  While the `ImagePreprocessor.estimate_skew_angle()` routine corrects tilt within $\pm 35^\circ$, 90-degree or 180-degree upside-down scans require explicit Tesseract OSD (Orientation and Script Detection) execution.

---

## 3. Multilingual / Non-Latin Scripts
- The present pipeline is optimized for Latin script invoices (English, European languages) and standard Indian GSTIN formats.
- Invoices in Devanagari, Arabic, or CJK characters require corresponding Tesseract language traineddata files (e.g. `hin.traineddata`, `chi_sim.traineddata`) and localized regex tokenizers.

---

## 4. Hand-Written Invoices
- Traditional OCR engines (Tesseract) and sequence CRFs trained on printed typography have low recognition rates for cursive or variable human handwriting.

---

## 5. Cold-Start / Untrained Mode Disclosures
- When no trained CRF model is loaded in the `models/` folder, the system falls back to regex candidate heuristics.
- The pipeline explicitly discloses this condition in the output metadata (`extraction_method: "rule_based_fallback"`) and attaches a validation warning to ensure human-in-the-loop review.
