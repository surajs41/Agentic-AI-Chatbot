THEMES = {
    "dark": {
        "label": "Dark",
        "bg": "#0b1120",
        "bg2": "#151d2e",
        "sidebar": "#111827",
        "text": "#f1f5f9",
        "muted": "#94a3b8",
        "accent": "#6366f1",
        "border": "rgba(148,163,184,0.2)",
        "card": "#151d2e",
        "input_bg": "#1e293b",
        "btn_secondary_bg": "#1e293b",
        "btn_secondary_text": "#f1f5f9",
    },
    "light": {
        "label": "Light",
        "bg": "#f1f5f9",
        "bg2": "#ffffff",
        "sidebar": "#ffffff",
        "text": "#0f172a",
        "muted": "#475569",
        "accent": "#4f46e5",
        "border": "rgba(15,23,42,0.12)",
        "card": "#ffffff",
        "input_bg": "#ffffff",
        "btn_secondary_bg": "#e2e8f0",
        "btn_secondary_text": "#0f172a",
    },
}


def build_css(theme_name: str) -> str:
    if theme_name not in THEMES:
        theme_name = "dark"
    t = THEMES[theme_name]

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

.stApp, [data-testid="stAppViewContainer"], .main, .main .block-container {{
    background: {t["bg"]} !important;
    color: {t["text"]} !important;
}}

[data-testid="stHeader"], [data-testid="stToolbar"], header {{
    background: {t["bg"]} !important;
}}

.stApp p, .stApp li, .stApp span, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {{
    color: {t["text"]} !important;
}}
[data-testid="stCaptionContainer"] {{ color: {t["muted"]} !important; }}

#MainMenu, footer {{ visibility: hidden; }}
.main .block-container {{ padding-top: 1rem; padding-bottom: 5rem; max-width: 1100px; }}

/* Buttons */
.stButton > button {{
    border-radius: 8px !important; font-weight: 500 !important;
    border: 1px solid {t["border"]} !important;
}}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {{
    background: {t["btn_secondary_bg"]} !important;
    color: {t["btn_secondary_text"]} !important;
}}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: {t["accent"]} !important;
    color: #ffffff !important;
    border: none !important;
}}

/* Popover — three dots only, hide chevron */
[data-testid="stSidebar"] [data-testid="stPopover"] > button {{
    background: {t["btn_secondary_bg"]} !important;
    color: {t["btn_secondary_text"]} !important;
    min-width: 2rem !important;
    padding: 0.25rem 0.4rem !important;
}}
[data-testid="stSidebar"] [data-testid="stPopover"] > button svg {{
    display: none !important;
}}

/* Selectbox / dropdown */
div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background: {t["input_bg"]} !important;
    color: {t["text"]} !important;
    border-color: {t["border"]} !important;
}}
[data-testid="stSelectbox"] label {{ color: {t["text"]} !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {t["sidebar"]} !important;
    border-right: 1px solid {t["border"]};
}}
[data-testid="stSidebar"] * {{ color: {t["text"]} !important; }}
[data-testid="stSidebar"] .stCaption {{ color: {t["muted"]} !important; }}
[data-testid="stSidebar"] label[data-baseweb="radio"] {{
    background: {t["card"]}; border: 1px solid {t["border"]};
    border-radius: 8px; padding: 0.4rem 0.6rem; margin-bottom: 0.25rem;
}}

/* File uploader */
[data-testid="stFileUploader"], [data-testid="stFileUploader"] section {{
    background: {t["bg2"]} !important;
    border: 1px dashed {t["border"]} !important;
    border-radius: 8px !important;
}}
[data-testid="stFileUploader"] * {{ color: {t["text"]} !important; }}
[data-testid="stFileUploader"] small {{ color: {t["muted"]} !important; }}
[data-testid="stFileUploader"] button {{
    background: {t["accent"]} !important; color: #fff !important;
}}

/* Alerts */
[data-testid="stAlert"], .stAlert {{
    background: {t["card"]} !important;
    color: {t["text"]} !important;
    border: 1px solid {t["border"]} !important;
}}
[data-testid="stAlert"] * {{ color: {t["text"]} !important; }}

/* Cards */
.top-nav {{
    background: {t["card"]} !important;
    border: 1px solid {t["border"]};
    border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.75rem;
    color: {t["text"]} !important;
}}
.top-nav strong {{ color: {t["text"]} !important; }}
.top-nav span {{ color: {t["muted"]} !important; }}

.hero-wrap {{
    background: {t["card"]}; border: 1px solid {t["border"]};
    border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}}
.hero-wrap h1 {{ color: {t["text"]} !important; font-size: 1.4rem; margin: 0 0 0.35rem; }}
.hero-wrap p {{ color: {t["muted"]} !important; font-size: 0.9rem; }}

/* Chat */
[data-testid="stChatMessage"] {{
    background: {t["card"]} !important;
    border: 1px solid {t["border"]} !important;
    border-radius: 12px !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
    color: {t["text"]} !important;
}}
.msg-meta {{ color: {t["muted"]} !important; font-size: 0.75rem; margin-bottom: 0.3rem; }}

div[data-testid="stChatInput"] textarea {{
    background: {t["input_bg"]} !important; color: {t["text"]} !important;
    caret-color: {t["text"]} !important; border: 1px solid {t["border"]} !important;
    border-radius: 10px !important;
}}
div[data-testid="stChatInput"] textarea::placeholder {{ color: {t["muted"]} !important; }}

/* Right panel */
.info-panel {{
    background: {t["card"]}; border: 1px solid {t["border"]};
    border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem;
}}
.info-panel h4 {{ color: {t["text"]} !important; margin: 0 0 0.75rem; font-size: 0.95rem; }}
.info-row {{
    display: flex; justify-content: space-between;
    padding: 0.4rem 0; border-bottom: 1px solid {t["border"]};
    font-size: 0.84rem;
}}
.info-row span {{ color: {t["muted"]} !important; }}
.info-row strong {{ color: {t["text"]} !important; }}

.source-card {{
    background: {t["bg2"]}; border: 1px solid {t["border"]};
    border-left: 3px solid {t["accent"]}; border-radius: 8px;
    padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;
}}
.source-card p {{ color: {t["text"]} !important; margin: 0; font-size: 0.82rem; }}

.stTabs [data-baseweb="tab"] {{ color: {t["muted"]} !important; }}
.stTabs [aria-selected="true"] {{ color: {t["text"]} !important; }}

/* Dataframe / table */
[data-testid="stDataFrame"] {{ background: {t["card"]} !important; }}

.upload-row {{
    background: {t["card"]}; border: 1px solid {t["border"]};
    border-radius: 10px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;
}}
</style>
"""
