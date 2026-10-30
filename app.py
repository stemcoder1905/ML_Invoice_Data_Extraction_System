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


PIPELINE_BUILD_VERSION = "2026.10.30.v5_exact_party_isolation"


@st.cache_resource(show_spinner="Initializing ML & OCR Pipeline...")
def get_pipeline(build_version: str = PIPELINE_BUILD_VERSION):
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

if st.sidebar.button("🔄 Reload Pipeline / Clear Cache", help="Click to reload the latest model and processing pipeline."):
    st.cache_resource.clear()
    st.rerun()

active_tab = st.sidebar.radio(
    "Navigation",
    ["Single Invoice Extraction", "Batch Processing"],
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
            "Upload an invoice document or ZIP archive",
            type=["pdf", "png", "jpg", "jpeg", "zip"],
            help="Supports digital PDFs, scanned PDFs, raster images, and ZIP archives containing invoices.",
        )
    with col_sample:
        st.write("**Or select a demonstration sample:**")
        sample_files = list(INVOICES_DIR.glob("*.pdf")) + list(INVOICES_DIR.glob("*.png"))
        sample_names = ["-- None --"] + [f.name for f in sample_files[:10]]
        selected_sample = st.selectbox("Sample Invoices", sample_names)

    selected_bytes = None
    selected_name = "invoice"

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".zip"):
            extracted_items = DocumentLoader.extract_zip(file_bytes)
            if not extracted_items:
                st.error("The uploaded ZIP archive does not contain any supported invoice files (.pdf, .png, .jpg, .jpeg).")
            else:
                st.info(f"📦 Archive contains {len(extracted_items)} invoice(s).")
                zip_choice = st.selectbox(
                    "Select invoice from ZIP archive to extract:",
                    [item[0] for item in extracted_items],
                    index=0,
                )
                for item_name, item_bytes in extracted_items:
                    if item_name == zip_choice:
                        selected_name = item_name
                        selected_bytes = item_bytes
                        break
        else:
            selected_bytes = file_bytes
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
        "Upload multiple invoices or ZIP archives (PDF, Images, ZIP)",
        type=["pdf", "png", "jpg", "jpeg", "zip"],
        accept_multiple_files=True,
    )

    run_sample_batch = st.button("🚀 Or Run Batch on Demonstration Invoices Folder")

    files_to_run = []
    if uploaded_files:
        for f in uploaded_files:
            f_bytes = f.getvalue()
            if f.name.lower().endswith(".zip"):
                extracted = DocumentLoader.extract_zip(f_bytes)
                if extracted:
                    st.info(f"📦 Extracted {len(extracted)} invoice(s) from ZIP archive: **{f.name}**")
                    for sub_name, sub_bytes in extracted:
                        files_to_run.append((sub_name, sub_bytes))
                else:
                    st.warning(f"⚠️ ZIP archive '{f.name}' contained no supported invoice files.")
            else:
                files_to_run.append((f.name, f_bytes))
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
                    "validation": {
                        "is_valid": False,
                        "errors": [f"Processing Exception: {str(e)}"],
                        "warnings": [str(e)],
                    },
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

