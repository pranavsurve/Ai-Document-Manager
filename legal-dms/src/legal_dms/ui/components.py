from __future__ import annotations

import streamlit as st
from typing import Iterable

from legal_dms.common import ui_theme


def sidebar_nav(selected: str = "library") -> str:
    """Render the narrow sidebar navigation and return the selected view.

    The sidebar contains three text-only navigation items: Library, Review, Chat.
    Selection is persisted in `st.session_state['ldms_view']`.
    """
    if 'ldms_view' not in st.session_state:
        st.session_state['ldms_view'] = selected

    # Use raw markdown links with classes for styling
    options = [("library", "Library"), ("review", "Review"), ("chat", "Chat")]
    for key, label in options:
        is_selected = st.session_state['ldms_view'] == key
        css_class = "ldms-sidebar-item selected" if is_selected else "ldms-sidebar-item"
        if st.button(label, key=f"nav_{key}"):
            st.session_state['ldms_view'] = key
            is_selected = True
        # render label with style (we still rely on button for keyboard focus/accessibility)
        st.markdown(f"<div class='{css_class}'>{label}</div>", unsafe_allow_html=True)

    return st.session_state['ldms_view']


def card(content_callable, key: str | None = None) -> None:
    """Render a simple card container. `content_callable` is called inside the card.
    """
    with st.container():
        st.markdown("<div class='ldms-card'>", unsafe_allow_html=True)
        content_callable()
        st.markdown("</div>", unsafe_allow_html=True)


def table_row(cells: Iterable[str], row_key: str, on_click=None) -> None:
    """Render a single table row. `on_click` can be a callable executed when the row button is pressed.
    We render a button to make the row keyboard-accessible.
    """
    cols = st.columns([2, 2, 3, 1, 1])
    for idx, col in enumerate(cols):
        try:
            value = list(cells)[idx]
        except Exception:
            value = ""
        if idx == 4:
            if on_click:
                if col.button(value, key=f"rowbtn_{row_key}"):
                    on_click()
            else:
                col.write(value)
        else:
            col.write(value)
