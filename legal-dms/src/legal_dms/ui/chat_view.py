from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from typing import List

from legal_dms.chat import answer_stream, answer
from legal_dms.indexer import search
from legal_dms.config.settings import settings
from legal_dms.common import ui_theme
from .components import card


def render_chat():
    st.markdown("<div class='ldms-title'>Chat</div>", unsafe_allow_html=True)
    if 'ldms_chat_history' not in st.session_state:
        st.session_state['ldms_chat_history'] = []

    # helper to get document choices for the selector
    def _get_document_choices():
        library = settings.library_path
        choices = [("All documents", None)]  # (display, document_id)
        for root, _, files in os.walk(library):
            for fname in files:
                if fname.endswith(".json"):
                    sidecar = Path(root) / fname
                    try:
                        data = json.loads(sidecar.read_text(encoding="utf-8"))
                        doc_id = data.get("document_id")
                        original_filename = data.get("original_filename") or sidecar.stem
                        meta = data.get("metadata") or {}
                        doc_type = meta.get("document_type", "unknown")
                        display = f"{original_filename} ({doc_type})"
                        choices.append((display, doc_id))
                    except Exception:
                        # skip unreadable sidecar
                        continue
        return choices

    # document selector
    doc_choices = _get_document_choices()
    doc_options = [display for display, _ in doc_choices]
    doc_id_map = {display: doc_id for display, doc_id in doc_choices}
    doc_choice = st.selectbox("Document", options=doc_options, index=0)
    selected_doc_id = doc_id_map[doc_choice]

    # Clear conversation link
    if st.checkbox("Clear conversation", key="clear_chat"):
        st.session_state['ldms_chat_history'] = []

    # render conversation
    container = st.container()
    for turn in st.session_state['ldms_chat_history']:
        if turn['role'] == 'user':
            container.markdown(f"<div class='ldms-chat-user'>{turn['text']}</div>", unsafe_allow_html=True)
        else:
            container.markdown(f"<div class='ldms-chat-assistant'>{turn['text']}</div>", unsafe_allow_html=True)
            if 'sources' in turn and turn['sources']:
                container.markdown("<div class='ldms-sources'>Sources:</div>", unsafe_allow_html=True)
                for s in turn['sources']:
                    container.markdown(f"<div class='ldms-small'>{s.document_id[:8]} p.{s.page_number}</div>", unsafe_allow_html=True)

    # input pinned to bottom: simple form
    with st.form(key="chat_input", clear_on_submit=False):
        user_text = st.text_area("Your question", height=120)
        submit = st.form_submit_button("Send")
        if submit and user_text.strip():
            # append user turn
            st.session_state['ldms_chat_history'].append({'role': 'user', 'text': user_text})
            placeholder = st.empty()
            tokens = []
            with st.spinner("Assistant is typing…"):
                for frag in answer_stream(user_text, document_id=selected_doc_id):
                    if isinstance(frag, str):
                        tokens.append(frag)
                        placeholder.markdown(f"<div class='ldms-chat-assistant'>{' '.join(tokens)}</div>", unsafe_allow_html=True)
                    elif isinstance(frag, dict) and frag.get('type') == 'sources':
                        sources = frag.get('sources', [])
                        st.session_state['ldms_chat_history'].append({'role': 'assistant', 'text': ' '.join(tokens), 'sources': sources})
                        placeholder.empty()
                        break
