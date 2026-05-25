from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from legal_dms.config.settings import settings
from legal_dms.pipeline import PipelineJob, Pipeline
from legal_dms.classifier.model import DocumentMetadata, DocumentType, Party
from legal_dms.common import ui_theme
from .components import card


def _list_pending_jobs() -> list[PipelineJob]:
    jobs: list[PipelineJob] = []
    for p in settings.work_path.iterdir():
        job_file = p / "job.json"
        if job_file.exists():
            try:
                job = PipelineJob.load(p.name)
                if job and job.current_stage == "await_confirmation":
                    jobs.append(job)
            except Exception:
                continue
    return jobs


def render_review():
    st.markdown("<div class='ldms-title'>Review</div>", unsafe_allow_html=True)
    pending = _list_pending_jobs()
    st.markdown(f"<div class='ldms-small'>{len(pending)} awaiting your decision</div>", unsafe_allow_html=True)

    for job in pending:
        def render_job_card(j=job):
            meta_dict = j.artifacts.get("metadata") or {}
            # suggested new name (simple): date-type-parties
            exec_date = meta_dict.get("execution_date", "")
            doc_type = meta_dict.get("document_type", "")
            parties = ", ".join([p.get("name") for p in meta_dict.get("parties", [])]) if meta_dict.get("parties") else ""
            st.markdown(f"<div class='ldms-body'>{doc_type} {parties}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='ldms-small'>{j.source_path.name}</div>", unsafe_allow_html=True)

            cols = st.columns([1, 2])
            # left: metadata
            left = cols[0]
            right = cols[1]
            left.markdown("<div class='ldms-small'>Metadata</div>", unsafe_allow_html=True)
            left.write(json.dumps(meta_dict, indent=2))

            right.markdown("<div class='ldms-small'>OCR excerpt</div>", unsafe_allow_html=True)
            ocr = j.artifacts.get("ocr_result") or {}
            excerpt = ""
            if isinstance(ocr, dict):
                pages = ocr.get("pages", [])
                if pages:
                    excerpt = pages[0].get("text", "")[:1500]
            right.write(excerpt)

            # actions
            action_cols = st.columns([1, 1, 1])
            if action_cols[0].button("Confirm", key=f"confirm_{j.document_id}"):
                # resume pipeline with an auto-confirming callback so it proceeds
                with st.spinner("Confirming and executing…"):
                    pipeline = Pipeline(confirmation_callback=lambda job: True)
                    try:
                        pipeline.process(j.source_path, resume_from_document_id=j.document_id)
                        st.success("Confirmed and executed.")
                    except Exception as e:
                        st.error(f"Failed to execute: {e}")
            if action_cols[1].button("Edit", key=f"edit_{j.document_id}"):
                # open a simple edit form
                with st.form(key=f"form_{j.document_id}"):
                    new_type = st.selectbox("Document type", options=[t.value for t in DocumentType], index=0)
                    new_parties = st.text_area("Parties (comma separated)")
                    new_date = st.text_input("Execution date (YYYY-MM-DD)")
                    submitted = st.form_submit_button("Apply")
                    if submitted:
                        # update metadata in job and save
                        try:
                            meta = j.artifacts.get("metadata") or {}
                            meta["document_type"] = new_type
                            meta["parties"] = [{"name": n.strip(), "role": "party"} for n in new_parties.split(",") if n.strip()]
                            if new_date:
                                meta["execution_date"] = new_date
                            j.artifacts["metadata"] = meta
                            j.save()
                            st.success("Updated metadata for review.")
                        except Exception as e:
                            st.error(f"Failed to save edits: {e}")
            if action_cols[2].button("Reject", key=f"reject_{j.document_id}"):
                # mark error on job
                j.error = "Rejected by reviewer"
                j.save()
                st.info("Item rejected.")

        card(render_job_card, key=f"card_{job.document_id}")
