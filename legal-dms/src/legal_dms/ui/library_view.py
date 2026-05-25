from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from legal_dms.config.settings import settings
from legal_dms.classifier.model import DocumentMetadata, Party
from legal_dms.indexer import remove_document, index_document
from legal_dms.common import ui_theme
from .components import card


LIB_COLS = ["Date", "Type", "Parties", "Language", "Open"]


def _load_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_library() -> list[dict[str, Any]]:
    library = settings.library_path
    results: list[dict[str, Any]] = []
    for root, _, files in os.walk(library):
        for fname in files:
            if fname.endswith(".json"):
                sidecar = Path(root) / fname
                data = _load_sidecar(sidecar)
                if not data:
                    continue
                meta = data.get("metadata")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = None
                results.append({
                    "sidecar": sidecar,
                    "document_id": data.get("document_id"),
                    "original_filename": data.get("original_filename") or sidecar.stem,
                    "metadata": meta,
                    "ocr_text_path": data.get("ocr_text_path"),  # kept for backward compatibility
                    "ocr_text": data.get("ocr_text"),
                    "destination_path": data.get("destination_path"),
                })
    return results


def _open_with_os(path: str) -> None:
    try:
        if os.name == 'nt':
            os.startfile(path)
        else:
            import subprocess

            subprocess.Popen(["xdg-open", path])
    except Exception:
        st.warning("Could not open file with the OS default viewer.")


def _make_ocr_result_from_text(text: str, language: str | None):
    class Page:
        def __init__(self, page_number: int, text: str, language: str | None = None):
            self.page_number = page_number
            self.text = text
            self.language = language or (language or "en")

    class FakeOcr:
        def __init__(self, text: str, language: str | None = None):
            self.pages = [Page(1, text, language)]

    return FakeOcr(text, language)


def render_library():
    st.markdown("<div class='ldms-title'>Library</div>", unsafe_allow_html=True)
    cols = st.columns([3, 2, 4, 2])
    # Filters
    doc_types = [t.value for t in __import__("legal_dms.classifier.model", fromlist=["DocumentType"]).DocumentType]
    selected_types = cols[0].multiselect("Type", options=doc_types, default=None)
    all_years = sorted({p.parent.name for p in settings.library_path.rglob("*") if p.is_file()}, reverse=True)
    selected_year = cols[1].selectbox("Year", options=["All"] + all_years, index=0)
    query = cols[2].text_input("Search")

    rows = _list_library()

    # simple filtering
    def matches(r: dict[str, Any]) -> bool:
        meta = r.get("metadata") or {}
        if selected_types:
            if meta.get("document_type") not in selected_types:
                return False
        if selected_year and selected_year != "All":
            exec_date = meta.get("execution_date")
            if exec_date:
                # exec_date is a string in ISO format, we can extract the year
                year = exec_date.split("-")[0] if exec_date else ""
                if selected_year != year:
                    return False
            else:
                return False
        if query:
            q = query.lower()
            if q not in r.get("original_filename", "").lower() and q not in json.dumps(meta or {}).lower():
                return False
        return True

    filtered = [r for r in rows if matches(r)]

    # Table header
    header_cols = st.columns([2, 2, 3, 1, 1])
    for c, label in zip(header_cols, LIB_COLS):
        c.markdown(f"**{label}**")

    for item in filtered:
        meta = item.get("metadata") or {}
        date_str = meta.get("execution_date", "")
        display_date = date_str.split("T")[0] if date_str else ""
        parties = ", ".join([p.get("name") for p in meta.get("parties", [])]) if meta.get("parties") else ""
        language = meta.get("primary_language", "")
        cols = st.columns([2, 2, 3, 1, 1])
        cols[0].write(display_date)
        cols[1].write(meta.get("document_type", ""))
        cols[2].write(parties)
        cols[3].write(language)
        if cols[4].button("View OCR", key=f"open_{item.get('document_id')}"):
            ocr_text = item.get("ocr_text")
            if ocr_text:
                with st.expander("OCR Text", expanded=True):
                    st.text_area("OCR Text", value=ocr_text, height=300, label_visibility="collapsed")
            else:
                st.warning("OCR text not available for this document.")

        # Row click to expand details
        if st.button("Details", key=f"details_{item.get('document_id')}"):
            with st.container():
                st.markdown("<div style='padding:24px'>", unsafe_allow_html=True)
                st.markdown("<div class='ldms-body'><strong>Summary</strong></div>", unsafe_allow_html=True)
                st.write(meta.get("summary", "No summary available."))
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                st.markdown("<div class='ldms-body'><strong>Key clauses</strong></div>", unsafe_allow_html=True)
                for clause in meta.get("key_clauses", []) or []:
                    st.write(f"- {clause}")

                if st.button("Re-index", key=f"reindex_{item.get('document_id')}"):
                    # remove from index and re-add from sidecar OCR text if available
                    with st.spinner("Re-indexing…"):
                        remove_document(item.get("document_id"))
                        ocr_text = item.get("ocr_text")
                        meta_data = meta
                        if ocr_text:
                            fake_ocr = _make_ocr_result_from_text(ocr_text, meta_data.get("primary_language"))
                            try:
                                # build DocumentMetadata model for indexer
                                doc_meta = DocumentMetadata(**meta_data)
                                index_document(item.get("document_id"), fake_ocr, doc_meta)
                                st.success("Re-indexed successfully.")
                            except Exception as e:
                                st.error(f"Failed to re-index: {e}")
                        else:
                            st.error("No OCR text available to re-index.")

                st.markdown("</div>", unsafe_allow_html=True)

