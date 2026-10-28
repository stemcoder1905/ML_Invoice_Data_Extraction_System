"""
Streamlit Web Application for ML-Based Invoice Data Extraction System.
Supports single and batch invoice uploads (PDF, PNG, JPG, JPEG), visual preview,
real-time CRF inference, interactive manual field editing, JSON download,
validation diagnostics, and held-out evaluation reports.
"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import io
import zipfile

from src.config import TARGET_FIELDS, INVOICES_DIR, DEFAULT_MODEL_PATH, SUPPORTED_EXTENSIONS
from src.document_loader import DocumentLoader
from src.ocr_engine import OCREngine
from src.postprocessor import InvoiceProcessor
from src.ml_extractor import InvoiceCRFExtractor
from src.exporter import InvoiceJSONExporter, sanitize_for_json
from src.batch_processor import BatchProcessor
from evaluate import evaluate_test_set


st.set_page_config(
    page_title="ML Invoice Data Extraction System",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-ml {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
    }
    .badge-ocr {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    """Initializes and caches the invoice processing pipeline."""
    loader = DocumentLoader()
    ocr = OCREngine()
    crf = InvoiceCRFExtractor()
    processor = InvoiceProcessor(crf_extractor=crf)
    exporter = InvoiceJSONExporter()
    return loader, ocr, crf, processor, exporter


loader, ocr_engine, crf_model, processor, exporter = get_pipeline()

# Sidebar
st.sidebar.title("🧾 Control Panel")
st.sidebar.markdown(
    """
    **Architecture:** Traditional ML Pipeline  
    **Algorithm:** Conditional Random Fields (CRF)  
    **Text Extraction:** PyMuPDF + Tesseract  
    **Zero-LLM Guarantee:** 100% Traditional ML  
    """
)

model_status = "✅ Trained & Active" if crf_model.is_trained() else "⚠️ Untrained (Fallback Mode)"
st.sidebar.info(f"**Model Status:** {model_status}")

active_tab = st.sidebar.radio(
    "Navigation",
    ["Single Invoice Extraction", "Batch Processing", "Model Evaluation", "Architecture & Docs"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>ML Intern Assessment Project<br>Production-Oriented Invoice Extraction</small>",
    unsafe_allow_html=True,
)

# Tab 1: Single Invoice Extraction
if active_tab == "Single Invoice Extraction":
    st.markdown('<div class="main-header">Single Invoice Data Extraction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div><span class="badge-ml">Traditional ML: CRF Sequence Labeling</span>'
        '<span class="badge-ocr">PyMuPDF + OpenCV Preprocessing</span></div><br>',
        unsafe_allow_html=True,
    )

    col_upload, col_sample = st.columns([3, 2])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload an invoice document",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Supports digital PDFs, scanned PDFs, and raster images.",
        )
    with col_sample:
        st.write("**Or select a demonstration sample:**")
        sample_files = list(INVOICES_DIR.glob("*.pdf")) + list(INVOICES_DIR.glob("*.png"))
        sample_names = ["-- None --"] + [f.name for f in sample_files[:10]]
        selected_sample = st.selectbox("Sample Invoices", sample_names)

    selected_bytes = None
    selected_name = "invoice"

    if uploaded_file is not None:
        selected_bytes = uploaded_file.read()
        selected_name = uploaded_file.name
    elif selected_sample != "-- None --":
        sample_path = INVOICES_DIR / selected_sample
        selected_bytes = sample_path.read_bytes()
        selected_name = selected_sample

    if selected_bytes:
        with st.spinner("Processing invoice layout, OCR coordinates, and CRF entity tags..."):
            start_proc = time.time()
            doc_dict = loader.load(selected_bytes, filename=selected_name)
            tokens = ocr_engine.extract_tokens(doc_dict)
            extraction_result = processor.process(doc_dict, tokens)
            proc_time = round((time.time() - start_proc) * 1000, 1)

        st.success(f"Extracted {len(tokens)} tokens across {doc_dict['page_count']} page(s) in {proc_time} ms.")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📄 Document Preview")
            if doc_dict["pages"]:
                first_page_img = doc_dict["pages"][0]["image"]
                # Convert BGR to RGB for Streamlit display
                rgb_img = first_page_img[:, :, ::-1]
                st.image(rgb_img, caption=f"Page 1 - {selected_name}", use_container_width=True)

            with st.expander("🔍 Processing Metadata & Provenance", expanded=False):
                st.json(extraction_result["metadata"])

            with st.expander("⚠️ Validation Warnings & Diagnostics", expanded=True):
                val_info = extraction_result["validation"]
                if val_info["is_valid"]:
                    st.success("✅ Document validation passed.")
                else:
                    st.warning("⚠️ Validation warnings flagged.")

                if val_info["warnings"]:
                    for w in val_info["warnings"]:
                        st.markdown(f"- ⚠️ {w}")
                else:
                    st.markdown("*No warnings detected.*")

        with col_right:
            st.subheader("📝 Extracted Header Fields (Editable)")
            st.markdown("<small>Review and manually correct extracted values before export:</small>", unsafe_allow_html=True)

            fields = extraction_result["fields"]
            with st.form("edit_invoice_form"):
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    e_inv_num = st.text_input("Invoice Number", value=str(fields.get("invoice_number") or ""))
                    e_inv_date = st.text_input("Invoice Date", value=str(fields.get("invoice_date") or ""))
                    e_supplier = st.text_input("Supplier / Vendor Name", value=str(fields.get("supplier_name") or ""))
                    e_buyer = st.text_input("Buyer / Customer Name", value=str(fields.get("buyer_name") or ""))
                    e_po = st.text_input("Purchase Order Number", value=str(fields.get("purchase_order_number") or ""))

                with f_col2:
                    e_currency = st.text_input("Currency", value=str(fields.get("currency") or ""))
                    e_tax_id = st.text_input("Tax ID / GSTIN", value=str(fields.get("tax_id") or ""))
                    e_subtotal = st.text_input("Subtotal", value=str(fields.get("subtotal") or ""))
                    e_tax = st.text_input("Tax Amount", value=str(fields.get("tax_amount") or ""))
                    e_total = st.text_input("Total Amount", value=str(fields.get("total_amount") or ""))

                submit_edits = st.form_submit_button("Update Fields")

            if submit_edits:
                fields["invoice_number"] = e_inv_num
                fields["invoice_date"] = e_inv_date
                fields["supplier_name"] = e_supplier
                fields["buyer_name"] = e_buyer
                fields["purchase_order_number"] = e_po
                fields["currency"] = e_currency
                fields["tax_id"] = e_tax_id
                fields["subtotal"] = float(e_subtotal) if e_subtotal else None
                fields["tax_amount"] = float(e_tax) if e_tax else None
                fields["total_amount"] = float(e_total) if e_total else None
                st.info("Updated fields stored.")

            st.subheader("📊 Extracted Line Items")
            line_items = extraction_result.get("line_items", [])
            if line_items:
                df_items = pd.DataFrame(line_items)
                edited_df = st.data_editor(df_items, use_container_width=True, num_rows="dynamic")
                extraction_result["line_items"] = edited_df.to_dict(orient="records")
            else:
                st.info("No line item table rows detected.")

            st.subheader("💾 Export JSON")
            json_str = exporter.to_json_string(extraction_result)
            st.download_button(
                label="📥 Download Extracted JSON",
                data=json_str,
                file_name=f"{Path(selected_name).stem}_extracted.json",
                mime="application/json",
            )

            with st.expander("👁️ View Full JSON Payload"):
                st.code(json_str, language="json")

# Tab 2: Batch Processing
elif active_tab == "Batch Processing":
    st.markdown('<div class="main-header">Batch Invoice Processing</div>', unsafe_allow_html=True)
    st.markdown("Process multiple invoices concurrently with automated fault isolation and summary reporting.")

    uploaded_files = st.file_uploader(
        "Upload multiple invoices (PDF / Images)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    run_sample_batch = st.button("🚀 Or Run Batch on Demonstration Invoices Folder")

    files_to_run = []
    if uploaded_files:
        files_to_run = [(f.name, f.read()) for f in uploaded_files]
    elif run_sample_batch:
        demo_files = list(INVOICES_DIR.glob("*.pdf"))[:8]
        files_to_run = [(f.name, f.read_bytes()) for f in demo_files]

    if files_to_run:
        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_results = []
        start_batch = time.time()

        for idx, (fname, fbytes) in enumerate(files_to_run):
            status_text.text(f"Processing ({idx + 1}/{len(files_to_run)}): {fname}")
            try:
                doc = loader.load(fbytes, filename=fname)
                toks = ocr_engine.extract_tokens(doc)
                res = processor.process(doc, toks)
                batch_results.append(res)
            except Exception as e:
                batch_results.append({
                    "metadata": {"filename": fname, "error": str(e)},
                    "fields": {},
                    "line_items": [],
                    "validation": {"is_valid": False, "warnings": [str(e)]},
                })
            progress_bar.progress((idx + 1) / len(files_to_run))

        total_time = round(time.time() - start_batch, 2)
        status_text.text("Batch processing finished!")

        st.subheader("📈 Batch Processing Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Invoices", len(batch_results))
        m2.metric("Successful", sum(1 for r in batch_results if "error" not in r["metadata"]))
        m3.metric("Valid Invoices", sum(1 for r in batch_results if r["validation"].get("is_valid", False)))
        m4.metric("Total Time", f"{total_time} s")

        # Tabular overview
        rows = []
        for r in batch_results:
            f = r.get("fields", {})
            m = r.get("metadata", {})
            v = r.get("validation", {})
            rows.append({
                "Filename": m.get("filename", ""),
                "Invoice #": f.get("invoice_number", ""),
                "Date": f.get("invoice_date", ""),
                "Supplier": f.get("supplier_name", ""),
                "Buyer": f.get("buyer_name", ""),
                "Total Amount": f.get("total_amount", ""),
                "Status": "Valid" if v.get("is_valid", False) else "Warning",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ZIP download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for r in batch_results:
                fname = r.get("metadata", {}).get("filename", "invoice")
                stem = Path(fname).stem
                json_data = exporter.to_json_string(r)
                zip_file.writestr(f"{stem}_extracted.json", json_data)

        st.download_button(
            label="📦 Download All Extracted JSONs (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="batch_extracted_invoices.zip",
            mime="application/zip",
        )

# Tab 3: Model Evaluation
elif active_tab == "Model Evaluation":
    st.markdown('<div class="main-header">Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        "Strictly evaluated on a held-out test split of annotated invoices (zero token leakage). "
        "All metrics are calculated directly from ground-truth comparisons."
    )

    eval_report_path = Path("output") / "evaluation_report.json"

    if st.button("⚡ Re-run Evaluation on Held-Out Test Set"):
        with st.spinner("Running evaluation over test invoices..."):
            eval_data = evaluate_test_set(output_report_file=eval_report_path)
            st.success("Evaluation completed.")
    elif eval_report_path.exists():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    else:
        st.info("No existing evaluation report found. Click the button above to run evaluation.")
        eval_data = None

    if eval_data:
        summary = eval_data.get("overall_summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Micro Precision", f"{summary.get('micro_precision', 0.0)*100:.1f}%")
        c2.metric("Micro Recall", f"{summary.get('micro_recall', 0.0)*100:.1f}%")
        c3.metric("Micro F1-Score", f"{summary.get('micro_f1', 0.0)*100:.1f}%")
        c4.metric("Macro F1-Score", f"{summary.get('macro_f1', 0.0)*100:.1f}%")

        st.subheader("📋 Field-Level Metric Breakdown")
        field_breakdown = eval_data.get("field_breakdown", {})
        breakdown_rows = []
        for field, metrics in field_breakdown.items():
            breakdown_rows.append({
                "Field Name": field,
                "Precision": f"{metrics['precision']*100:.1f}%",
                "Recall": f"{metrics['recall']*100:.1f}%",
                "F1-Score": f"{metrics['f1_score']*100:.1f}%",
                "Exact Match Accuracy": f"{metrics['exact_match_accuracy']*100:.1f}%",
                "Support": metrics["support"],
            })
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)

        st.subheader("📑 Tabular Line-Item Extraction Metrics")
        li_metrics = eval_data.get("line_items_summary", {})
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Ground Truth Items", li_metrics.get("ground_truth_items", 0))
        l2.metric("Extracted Items", li_metrics.get("extracted_items", 0))
        l3.metric("Precision", f"{li_metrics.get('precision', 0.0)*100:.1f}%")
        l4.metric("Recall / F1", f"{li_metrics.get('f1', 0.0)*100:.1f}%")

# Tab 4: Architecture & Docs
elif active_tab == "Architecture & Docs":
    st.markdown('<div class="main-header">System Architecture & ML Methodology</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 🏛️ Pipeline Overview
        This invoice data extraction system strictly utilizes **Traditional Machine Learning & OCR** techniques.
        No Generative AI, Large Language Models (LLMs), or black-box vision APIs are utilized.
        """
    )

    st.markdown(
        """
        ```
        +-------------------------------------------------------------+
        |                    INVOICE DOCUMENT                         |
        |              (Digital PDF / Scanned PDF / Image)            |
        +-------------------------------------------------------------+
                                      |
                                      v
        +-------------------------------------------------------------+
        |                 OPENCV PREPROCESSING ENGINE                 |
        |   - Aspect-preserving resize                                |
        |   - Grayscale conversion                                    |
        |   - Gaussian / Bilateral noise filtering                    |
        |   - Skew angle detection (minAreaRect) & deskew rotation    |
        |   - Otsu & Adaptive binarization                            |
        +-------------------------------------------------------------+
                                      |
                                      v
        +-------------------------------------------------------------+
        |             OCR & WORD BOUNDING BOX EXTRACTION              |
        |   - Digital PDFs: PyMuPDF lossless native word tokens       |
        |   - Scanned Images: Tesseract OCR (pytesseract word boxes)  |
        |   - Normalized coordinates [0.0, 1.0] & natural reading order|
        +-------------------------------------------------------------+
                                      |
                                      v
        +-------------------------------------------------------------+
        |                 FEATURE ENGINEERING PIPELINE                |
        |   - Lexical: text lower, length, prefixes, suffixes         |
        |   - Orthographic: shape pattern (e.g. XXX-9999), isupper    |
        |   - Domain Lexicons: invoice, date, total, tax, vendor flags|
        |   - Positional: normalized x0, y0, page regions, bins       |
        |   - Contextual: -2, -1, +1, +2 window tokens & spatial gaps |
        +-------------------------------------------------------------+
                        |                             |
                        v                             v
        +-----------------------------+ +-----------------------------+
        |      TRADITIONAL ML:        | |   LINE ITEM TABLE EXTRACTOR |
        |   CONDITIONAL RANDOM FIELDS | | - Horizontal line grouping  |
        |   (CRF Sequence Tagging)    | | - Column header spans       |
        |   - BIO entity tagging      | | - Row coordinate clustering |
        |   - Model: lbfgs, c1, c2    | | - Arithmetic reconciliation |
        +-----------------------------+ +-----------------------------+
                        |                             |
                        +--------------+--------------+
                                       |
                                       v
        +-------------------------------------------------------------+
        |             HYBRID MERGE & RULE COMPLEMENTATION             |
        |   - Merges CRF entity spans with regex candidate extractors |
        |   - Discloses provenance (crf_ml / hybrid / fallback)       |
        +-------------------------------------------------------------+
                                       |
                                       v
        +-------------------------------------------------------------+
        |              FINANCIAL VALIDATION & QA ENGINE               |
        |   - GSTIN format checksum check                             |
        |   - Date parsing (ISO YYYY-MM-DD normalization)             |
        |   - Subtotal + Tax == Total mathematical consistency check  |
        |   - Line item sum reconciliation                            |
        +-------------------------------------------------------------+
                                       |
                                       v
        +-------------------------------------------------------------+
        |                    RFC-COMPLIANT JSON EXPORT                |
        |   - Standardized fields, line items, validation & metadata  |
        +-------------------------------------------------------------+
        ```
        """
    )
