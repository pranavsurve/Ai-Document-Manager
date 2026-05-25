from legal_dms.common.ui_theme import INK, MUTED, SURFACE, ACCENT, FONT_FAMILY


def get_base_css() -> str:
    """Return the single CSS block used to style the Streamlit UI according to the design system.

    Rules enforced:
    - Only uses the four design colors imported from common.ui_theme
    - Inter font (fallbacks retained)
    - Sizes: body 14px, small 12px, title 20px
    - Spacing tokens: 4,8,16,24,32,48
    - Sidebar width ~220px, max content width ~960px
    - Focus outline uses 2px ACCENT
    - Hides Streamlit default header/footer
    """

    css = f"""
    <style>
    :root {{
      --color-ink: {INK};
      --color-muted: {MUTED};
      --color-surface: {SURFACE};
      --color-accent: {ACCENT};
      --font-family: {FONT_FAMILY};
      --type-body: 14px;
      --type-small: 12px;
      --type-title: 20px;
      --space-xs: 4px;
      --space-sm: 8px;
      --space-md: 16px;
      --space-lg: 24px;
      --space-xl: 32px;
      --space-xxl: 48px;
      --sidebar-width: 220px;
      --max-content-width: 960px;
      --panel-radius: 6px;
    }}

    /* Reset streamlit chrome and font */
    .css-1d391kg {{ display: none !important; }} /* header */
    .css-1rs6os.edgvbvh3 {{ display: none !important; }} /* footer (varies across versions) */
    html, body, [data-testid="stAppViewContainer"] {{
      font-family: var(--font-family) !important;
      background: var(--color-surface) !important;
      color: var(--color-ink) !important;
      font-size: var(--type-body) !important;
    }}

    /* Layout centering and max width */
    .app-container, .block-container {{
      max-width: var(--max-content-width) !important;
      margin-left: auto !important;
      margin-right: auto !important;
      padding: var(--space-lg) !important;
    }}

    /* Sidebar sizing and simple nav look */
    aside[data-testid="stSidebar"] {{
      width: var(--sidebar-width) !important;
      min-width: var(--sidebar-width) !important;
      padding: var(--space-lg) !important;
      background: var(--color-surface) !important;
      border-right: 1px solid var(--color-muted);
    }}

    /* Buttons, inputs, textarea */
    .stButton>button, button, input[type="button"], input[type="submit"] {{
      border-radius: 4px !important;
      border: 1px solid transparent !important;
      background-color: var(--color-accent) !important;
      color: white !important;
      padding: 8px 12px !important;
      font-size: var(--type-body) !important;
      font-family: var(--font-family) !important;
    }}

    .stTextInput, .stTextArea, textarea, input[type="text"], select {{
      border: 1px solid var(--color-muted) !important;
      background: white !important;
      padding: 8px 10px !important;
      font-size: var(--type-body) !important;
      font-family: var(--font-family) !important;
      color: var(--color-ink) !important;
    }}

    /* Focus states */
    :focus, button:focus, input:focus, textarea:focus, select:focus {{
      outline: 2px solid var(--color-accent) !important;
      outline-offset: 2px !important;
    }}

    /* Text sizes */
    .ldms-title {{ font-size: var(--type-title) !important; font-weight: 600 !important; color: var(--color-ink) !important; }}
    .ldms-body {{ font-size: var(--type-body) !important; color: var(--color-ink) !important; }}
    .ldms-small {{ font-size: var(--type-small) !important; color: var(--color-muted) !important; }}

    /* Cards and panels */
    .ldms-card {{
      padding: var(--space-lg) !important;
      border: 1px solid var(--color-muted) !important;
      background: var(--color-surface) !important;
      border-radius: var(--panel-radius) !important;
      margin-bottom: var(--space-md) !important;
    }}

    /* Table rows - simple separators only */
    .ldms-table-row {{
      padding: 12px 8px !important;
      border-bottom: 1px solid var(--color-muted) !important;
      display: flex; align-items: center;
    }}

    .ldms-table-cell {{ flex: 1; padding: 0 var(--space-sm); font-size: var(--type-body); color: var(--color-ink); }}
    .ldms-table-cell.small {{ font-size: var(--type-small); color: var(--color-muted); }}

    .ldms-sidebar-item {{ padding: 8px 6px; color: var(--color-muted); display: block; text-decoration: none; }}
    .ldms-sidebar-item.selected {{ color: var(--color-ink); border-left: 2px solid var(--color-accent); padding-left: 12px; font-weight: 600; }}

    /* Chat bubbles */
    .ldms-chat-user {{ background: var(--color-surface); padding: 16px; margin: 8px 0; border-radius: 4px; color: var(--color-ink); text-align: right; }}
    .ldms-chat-assistant {{ padding: 0 0 8px 0; margin: 8px 0; color: var(--color-ink); }}
    .ldms-sources {{ color: var(--color-muted); font-size: var(--type-small); margin-top: 8px; }}

    /* Small muted link */
    .ldms-link-muted {{ color: var(--color-muted); font-size: var(--type-small); cursor: pointer; text-decoration: underline; }}

    </style>
    """

    return css
