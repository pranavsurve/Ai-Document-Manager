from __future__ import annotations

import streamlit as st
from typing import List

from legal_dms.chat import answer_stream, answer
from legal_dms.indexer import search
from legal_dms.common import ui_theme
from .components import card


def render_chat():
    st.markdown("<div class='ldms-title'>Chat</div>", unsafe_allow_html=True)
    if 'ldms_chat_history' not in st.session_state:
        st.session_state['ldms_chat_history'] = []

    # document selector
    docs = ["All documents"] + ["Recent"]
    doc_choice = st.selectbox("Document", options=docs, index=0)

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
                for frag in answer_stream(user_text, document_id=None):
                    if isinstance(frag, str):
                        tokens.append(frag)
                        placeholder.markdown(f"<div class='ldms-chat-assistant'>{' '.join(tokens)}</div>", unsafe_allow_html=True)
                    elif isinstance(frag, dict) and frag.get('type') == 'sources':
                        sources = frag.get('sources', [])
                        st.session_state['ldms_chat_history'].append({'role': 'assistant', 'text': ' '.join(tokens), 'sources': sources})
                        placeholder.empty()
                        break
