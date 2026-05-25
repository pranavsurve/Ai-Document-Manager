"""Minimalist Streamlit interface for Legal DMS, using shared UI tokens for colors, type, and spacing."""

import streamlit as st

from .styles import get_base_css
from .components import sidebar_nav
from .library_view import render_library
from .review_view import render_review
from .chat_view import render_chat


def main():
    # apply styles
    st.markdown(get_base_css(), unsafe_allow_html=True)

    # layout: sidebar + main
    st.set_page_config(layout="wide", page_title="Legal DMS")

    with st.sidebar:
        view = sidebar_nav()

    # Main area
    if view == 'library':
        render_library()
    elif view == 'review':
        render_review()
    else:
        render_chat()


if __name__ == '__main__':
    main()
