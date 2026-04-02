"""
FarmGenius — AI-Powered Smart Crop Recommendation System
Production-ready Streamlit app for Indian farmers.
"""

import os
import base64
import pickle
import warnings

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# CONSTANTS & DATA
# ═══════════════════════════════════════════════

WEATHER_API_KEY = "850ec0960167a55a828cc741add8c29a"
MODEL_PATH      = "crop_model.pkl"
LOGO_PATH       = "crop1.png"

CROP_LIST = ["banana", "coffee", "cotton", "jute",
             "maize", "mango", "pomegranate", "rice"]

CROP_IMAGES = {c: f"images/{c}.png" for c in CROP_LIST}

CROP_EMOJI = {
    "banana": "🍌", "coffee": "☕", "cotton": "🌿", "jute": "🌾",
    "maize": "🌽", "mango": "🥭", "pomegranate": "🍎", "rice": "🍚"
}

CROP_DESC = {
    "banana":      "Thrives in humid tropical regions with well-drained loamy soil.",
    "coffee":      "Grows best in cool highlands with rich, well-drained volcanic soil.",
    "cotton":      "Prefers deep black soil with good water retention in warm climate.",
    "jute":        "Loves alluvial soil and high humidity with warm temperatures.",
    "maize":       "Adaptable crop that grows well in fertile, well-drained soils.",
    "mango":       "Flourishes in deep, well-drained soil with tropical climate.",
    "pomegranate": "Drought-resistant fruit crop ideal for semi-arid regions.",
    "rice":        "Requires clayey soil with high water retention and humid climate.",
}

MSP = {
    "rice": 2369, "maize": 2400, "cotton": 7710, "jute": 5650,
    "banana": 1500, "mango": 3000, "pomegranate": 4000, "coffee": 6000,
}

GOVT_MSP_CROPS = {"rice", "maize", "cotton", "jute"}

NPK_REQ = {
    "rice":        {"N": 80,  "P": 40, "K": 40},
    "jute":        {"N": 50,  "P": 25, "K": 30},
    "maize":       {"N": 60,  "P": 40, "K": 40},
    "cotton":      {"N": 80,  "P": 35, "K": 35},
    "banana":      {"N": 100, "P": 50, "K": 50},
    "mango":       {"N": 60,  "P": 30, "K": 30},
    "coffee":      {"N": 40,  "P": 30, "K": 30},
    "pomegranate": {"N": 50,  "P": 25, "K": 25},
}

TIPS = [
    ("🧪", "Test Soil",      "Test your soil nutrients every season"),
    ("💧", "Save Water",     "Use drip irrigation to conserve water"),
    ("🌤", "Watch Weather",  "Monitor weather before sowing"),
    ("🌱", "Go Organic",     "Use organic compost regularly"),
    ("🤝", "Join FPO",       "Join FPO for better crop prices"),
]

SCHEMES = [
    {
        "icon": "💰",
        "tag": "Income Support",
        "name": "PM-KISAN",
        "full": "Pradhan Mantri Kisan Samman Nidhi",
        "desc": "Direct income support of ₹6,000/year in three equal instalments to all landholding farmer families across India.",
        "link": "https://pmkisan.gov.in",
    },
    {
        "icon": "🛡️",
        "tag": "Crop Insurance",
        "name": "PMFBY",
        "full": "Pradhan Mantri Fasal Bima Yojana",
        "desc": "Comprehensive crop insurance covering sowing risk, standing crop loss & post-harvest damage at very low premium rates.",
        "link": "https://pmfby.gov.in",
    },
    {
        "icon": "🧪",
        "tag": "Soil Health",
        "name": "Soil Health Card",
        "full": "Soil Health Card Scheme",
        "desc": "Free soil testing and personalised nutrient cards issued to farmers every two years to guide fertilizer use and improve yield.",
        "link": "https://soilhealth.dac.gov.in",
    },
    {
        "icon": "🏪",
        "tag": "Digital Market",
        "name": "e-NAM",
        "full": "National Agriculture Market",
        "desc": "Pan-India online trading portal connecting farmers, traders and buyers for transparent price discovery and better market access.",
        "link": "https://enam.gov.in",
    },
    {
        "icon": "💧",
        "tag": "Irrigation",
        "name": "PMKSY",
        "full": "Pradhan Mantri Krishi Sinchayee Yojana",
        "desc": "Ensures water to every field (Har Khet Ko Pani) and promotes efficient water use through micro-irrigation like drip and sprinkler.",
        "link": "https://pmksy.gov.in",
    },
]

# ═══════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_base64_image(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def fetch_weather(city: str):
    if not city.strip():
        return None, None
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city.strip()},IN&appid={WEATHER_API_KEY}&units=metric"
        )
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        if "main" in data:
            return round(data["main"]["temp"], 1), data["main"]["humidity"]
    except requests.exceptions.Timeout:
        st.error("Request timed out. Check your internet connection.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error("City not found. Try a different spelling.")
        else:
            st.error(f"API error: {e.response.status_code}")
    except Exception:
        st.error("Could not fetch weather. Check your connection.")
    return None, None


def fertilizer_advice(N: int, P: int, K: int) -> tuple:
    if N < 50:
        return "🌿 Apply Urea", "Nitrogen is low — Urea boosts healthy leaf & stem growth.", "#2e7d32"
    if P < 30:
        return "🪨 Apply Super Phosphate", "Phosphorus is deficient — Super Phosphate supports strong roots.", "#5d4037"
    if K < 30:
        return "⚗️ Apply Potash", "Potassium is low — Muriate of Potash improves crop quality.", "#1565c0"
    return "✅ Soil is Balanced", "Nutrient levels are optimal. Maintain with organic compost.", "#388e3c"


def suitability_score(N, P, K, crop):
    req = NPK_REQ[crop]

    def calc_score(actual, ideal):
        tolerance = ideal * 0.5
        diff = abs(actual - ideal)
        if diff <= tolerance:
            return 100 - (diff / tolerance) * 50
        else:
            return max(0, 50 - (diff / ideal) * 50)

    sN = calc_score(N, req["N"])
    sP = calc_score(P, req["P"])
    sK = calc_score(K, req["K"])
    return round((sN + sP + sK) / 3, 1)


def get_label(score):
    if score > 80:
        return "Excellent"
    elif score > 60:
        return "Good"
    elif score > 40:
        return "Moderate"
    else:
        return "Poor"


def safe_image(path: str, **kwargs) -> bool:
    if path and os.path.exists(path):
        try:
            st.image(path, **kwargs)
            return True
        except Exception:
            pass
    return False


def score_card(crop: str, score: float, other_score: float,
               border_color: str, bar_grad: str):
    em    = CROP_EMOJI.get(crop, "🌿")
    label = get_label(score)
    win   = score >= other_score
    sc    = "#40916c" if win else "#9e9e9e"
    badge = (
        "<div style='background:#e8f5e9;border-radius:50px;padding:4px 14px;"
        "font-size:12px;font-weight:700;color:#1b4332;display:inline-block;"
        "margin-top:10px;'>✅ BETTER MATCH</div>"
    ) if win else ""
    w = min(score, 100)
    st.markdown(f"""
    <div class="cmp-card" style="border-top:4px solid {border_color};">
        <div style="font-size:44px;margin-bottom:6px;">{em}</div>
        <div style="font-family:'Playfair Display',serif;font-size:22px;
                    font-weight:700;color:#1b4332;">{crop.capitalize()}</div>
        <div style="font-size:11px;color:#999;margin:8px 0 6px;
                    text-transform:uppercase;letter-spacing:.08em;">Suitability Score</div>
        <div style="font-family:'Playfair Display',serif;font-size:52px;
                    font-weight:800;color:{sc};line-height:1.1;">{score}</div>
        <div style="font-size:12px;color:#bbb;margin-bottom:10px;">out of 100</div>
        <div style="font-size:13px;color:#666;font-weight:600;">{label}</div>
        <div style="background:#e8f5e9;border-radius:50px;height:10px;margin:0 6px;">
            <div style="background:{bar_grad};width:{w}%;height:10px;border-radius:50px;"></div>
        </div>
        {badge}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════

for _k, _v in {"temp": None, "hum": None, "prediction": None}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="FarmGenius — Smart Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
# CSS  (ALL braces inside f-string are doubled)
# ═══════════════════════════════════════════════

logo_b64  = load_base64_image(LOGO_PATH)
logo_html = (f'<img class="logo" src="data:image/png;base64,{logo_b64}">'
             if logo_b64 else "")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {{
    --g-dark:  #1b4332; --g-mid:  #2d6a4f; --g-main: #40916c;
    --g-light: #74c69d; --g-pale: #d8f3dc;
    --e-dark:  #5c3d2e; --e-mid:  #8b5e3c; --e-light: #d4a96a;
    --cream: #fdf8f0;
    --sh-s: 0 4px 24px rgba(27,67,50,.10);
    --sh-c: 0 8px 40px rgba(27,67,50,.14);
    --r-lg:20px; --r-md:14px; --r-sm:10px;
    --ff-d:'Playfair Display',Georgia,serif;
    --ff-b:'DM Sans',sans-serif;
}}

html,body,[class*="css"] {{ font-family:var(--ff-b); color:#1a2e1a; }}

.stApp {{
    background:linear-gradient(160deg,#f0f7f4 0%,#fdf8f0 50%,#f5f0e8 100%);
    min-height:100vh;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,var(--g-dark) 0%,var(--g-mid) 100%) !important;
    border-right:none !important;
}}
[data-testid="stSidebar"] * {{ color:#e8f5e9 !important; font-family:var(--ff-b) !important; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] label {{
    color:#b7e4c7 !important; font-weight:600 !important;
}}
[data-testid="stSidebar"] .stTextInput input {{
    background: #ffffff !important;
    border: 1.5px solid rgba(255,255,255,.4) !important;
    border-radius: var(--r-sm) !important;
    color: #1a2e1a !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
}}
[data-testid="stSidebar"] .stTextInput input::placeholder {{
    color: #7a8a7a !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] .stTextInput input:focus {{
    border-color: #74c69d !important;
    box-shadow: 0 0 0 2px rgba(116,198,157,.25) !important;
    outline: none !important;
}}
[data-testid="stSidebar"] .stButton>button {{
    background:linear-gradient(135deg,var(--e-light),var(--e-mid)) !important;
    color:white !important; border:none !important;
    border-radius:var(--r-sm) !important; font-weight:600 !important;
    width:100%; padding:.65em 1em;
    box-shadow:0 4px 15px rgba(0,0,0,.2); transition:all .25s ease;
}}
[data-testid="stSidebar"] .stButton>button:hover {{
    transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,.3);
}}

/* ── Selectbox z-index ── */
div[data-baseweb="select"] {{
    position: relative !important;
    z-index: 100 !important;
}}
div[data-baseweb="select"] > div:first-child {{
    position: relative !important;
    z-index: 101 !important;
    pointer-events: auto !important;
    cursor: pointer !important;
    border-radius: var(--r-sm) !important;
    border-color: #c8dfc8 !important;
    background: white !important;
}}
div[data-baseweb="select"] > div:first-child:focus-within {{
    border-color: #40916c !important;
    box-shadow: 0 0 0 2px rgba(64,145,108,.15) !important;
}}
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {{
    z-index: 99999 !important;
    position: absolute !important;
    pointer-events: auto !important;
    visibility: visible !important;
    opacity: 1 !important;
    display: block !important;
}}
li[role="option"] {{
    pointer-events: auto !important;
    cursor: pointer !important;
}}

/* ── Hide tooltip leakage ── */
[role="slider"]::before,
[role="slider"]::after {{ display: none !important; }}
[data-testid="stTooltipIcon"] {{
    display: none !important; visibility: hidden !important; pointer-events: none !important;
}}
span[aria-live], div[aria-live], [data-testid="stAriaLive"] {{
    display: none !important; visibility: hidden !important;
}}
[data-testid="stSidebar"] span[role="tooltip"],
[data-testid="stSidebar"] div[role="tooltip"],
[data-testid="stSidebar"] [data-baseweb="tooltip"] {{
    display: none !important; visibility: hidden !important;
    opacity: 0 !important; pointer-events: none !important;
}}

/* ── Hero ── */
.hero {{
    background:linear-gradient(135deg,var(--g-dark) 0%,var(--g-mid) 60%,var(--e-dark) 100%);
    border-radius:var(--r-lg); padding:52px 48px 44px;
    margin-bottom:36px; position:relative; overflow:hidden; box-shadow:var(--sh-c);
}}
.hero::before {{
    content:""; position:absolute; top:-60px; right:-60px;
    width:260px; height:260px; border-radius:50%; background:rgba(255,255,255,.04);
}}
.hero::after {{
    content:""; position:absolute; bottom:-80px; left:-40px;
    width:320px; height:320px; border-radius:50%; background:rgba(116,198,157,.08);
}}
.hero-badge {{
    display:inline-block; background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.2); border-radius:50px;
    padding:5px 16px; font-size:13px; color:var(--g-light);
    font-weight:500; letter-spacing:.06em; margin-bottom:18px; text-transform:uppercase;
}}
.hero-title {{
    font-family:var(--ff-d); font-size:52px; font-weight:800;
    color:#fff; letter-spacing:-.5px; line-height:1.1; margin-bottom:10px;
}}
.hero-title span {{ color:var(--g-light); }}
.hero-sub {{ font-size:17px; color:rgba(255,255,255,.72); font-weight:300; }}

/* ── Section headers ── */
.sec-head {{
    font-family:var(--ff-d); font-size:28px; font-weight:700;
    color:var(--g-dark); margin-bottom:4px;
}}
.sec-sub {{ font-size:14px; color:#6b7c6b; margin-bottom:22px; }}

/* ── Divider ── */
.divider {{ display:flex; align-items:center; gap:16px; margin:40px 0 32px; }}
.divider::before,.divider::after {{
    content:""; flex:1; height:1px;
    background:linear-gradient(90deg,transparent,rgba(64,145,108,.3),transparent);
}}
.divider span {{ font-size:20px; }}

/* ── Weather card ── */
.w-card {{
    border-radius:var(--r-md); padding:22px 26px; color:white;
    box-shadow:var(--sh-c); display:flex; flex-direction:column; gap:4px;
    transition:transform .2s ease;
}}
.w-card:hover {{ transform:translateY(-3px); }}
.w-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.1em; opacity:.7; font-weight:500; }}
.w-value {{ font-family:var(--ff-d); font-size:38px; font-weight:700; line-height:1.1; }}
.w-icon {{ font-size:26px; margin-bottom:4px; }}

/* ── Main buttons ── */
div[data-testid="stButton"]>button {{
    background:linear-gradient(135deg,var(--g-main) 0%,var(--g-dark) 100%) !important;
    color:white !important; border:none !important; border-radius:50px !important;
    padding:.75em 2.5em !important; font-size:16px !important; font-weight:600 !important;
    letter-spacing:.04em; width:100%;
    box-shadow:0 6px 24px rgba(45,106,79,.35) !important;
    transition:all .28s cubic-bezier(.34,1.56,.64,1) !important;
}}
div[data-testid="stButton"]>button:hover {{
    transform:translateY(-3px) scale(1.02) !important;
    box-shadow:0 12px 32px rgba(45,106,79,.45) !important;
}}

/* ── Result card ── */
.result-card {{
    background:linear-gradient(135deg,var(--g-dark) 0%,var(--g-main) 100%);
    border-radius:var(--r-lg); padding:36px 40px; color:white;
    text-align:center; box-shadow:var(--sh-c); animation:slideUp .5s ease both;
}}
.result-name {{
    font-family:var(--ff-d); font-size:44px; font-weight:800;
    letter-spacing:2px; text-transform:uppercase; color:var(--g-light); margin-bottom:8px;
}}
.result-lbl {{
    font-size:14px; opacity:.75; font-weight:400;
    letter-spacing:.06em; text-transform:uppercase; margin-bottom:4px;
}}

/* ── Fertilizer card ── */
.fert-card {{
    border-radius:var(--r-md); padding:22px 26px; border-left:5px solid;
    background:rgba(255,255,255,.92); box-shadow:var(--sh-s);
    margin-top:16px; animation:slideUp .55s ease both;
}}
.fert-name {{ font-size:18px; font-weight:700; margin-bottom:6px; }}
.fert-desc {{ font-size:14px; color:#4a5e4a; line-height:1.6; }}

/* ── Pill ── */
.pill {{
    background:var(--g-pale); border-radius:50px; padding:8px 18px;
    font-size:13px; font-weight:600; color:var(--g-dark);
    border:1px solid rgba(64,145,108,.2); margin-bottom:10px;
    display:inline-block; width:100%;
}}

/* ── Progress bar ── */
.bar-wrap {{ margin-bottom:16px; }}
.bar-lbl {{
    display:flex; justify-content:space-between;
    font-size:13px; font-weight:600; color:var(--g-dark); margin-bottom:5px;
}}
.bar-bg {{ background:#e8f5e9; border-radius:50px; height:10px; overflow:hidden; }}
.bar-fill {{ height:10px; border-radius:50px; }}

/* ── Compare card ── */
.cmp-card {{
    background:white; border-radius:var(--r-lg); padding:28px 20px;
    text-align:center; box-shadow:var(--sh-c);
    border:1px solid rgba(64,145,108,.10);
}}

/* ── Crop detail card ── */
.crop-card {{
    background:white; border-radius:var(--r-lg); padding:32px;
    box-shadow:var(--sh-c); border:1px solid rgba(64,145,108,.12);
    animation:slideUp .4s ease both;
}}
.crop-card-name {{
    font-family:var(--ff-d); font-size:32px; font-weight:800;
    color:var(--g-dark); margin-bottom:4px;
}}
.price-badge {{
    display:inline-block; background:linear-gradient(135deg,#f9a825,#f57f17);
    color:white; border-radius:50px; padding:6px 20px; font-size:16px;
    font-weight:700; margin-bottom:18px; box-shadow:0 4px 12px rgba(245,127,23,.3);
}}
.npk-grid {{
    display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:14px;
}}
.npk-box {{
    background:var(--g-pale); border-radius:var(--r-sm); padding:14px;
    text-align:center; border:1px solid rgba(64,145,108,.15);
}}
.npk-label {{
    font-size:11px; text-transform:uppercase; letter-spacing:.08em;
    color:var(--g-mid); font-weight:600; margin-bottom:4px;
}}
.npk-val {{ font-family:var(--ff-d); font-size:24px; font-weight:700; color:var(--g-dark); }}
.crop-desc {{
    font-size:14px; color:#5a6e5a; line-height:1.7;
    margin-top:14px; padding-top:14px; border-top:1px solid #edf2ed;
}}

/* ── Tip card ── */
.tip {{
    background:white; border-radius:var(--r-md); padding:18px 12px;
    text-align:center; box-shadow:var(--sh-s);
    border:1px solid rgba(64,145,108,.10); height:100%;
    transition:transform .2s ease,box-shadow .2s ease;
}}
.tip:hover {{ transform:translateY(-4px); box-shadow:var(--sh-c); }}
.tip-icon {{ font-size:26px; margin-bottom:8px; }}
.tip-head {{ font-size:13px; font-weight:700; color:var(--g-dark); margin-bottom:4px; }}
.tip-text {{ font-size:11px; color:#6b7c6b; line-height:1.5; }}

/* ── Crop image wrapper ── */
.img-wrap img {{
    border-radius:var(--r-md); box-shadow:var(--sh-c);
    width:100%; object-fit:cover; transition:transform .3s ease;
}}
.img-wrap img:hover {{ transform:scale(1.03); }}

/* ── Logo ── */
.logo {{
    position:fixed; top:70px; right:20px; width:110px; z-index:999;
    border-radius:12px; box-shadow:0 6px 20px rgba(0,0,0,.2);
    transition:transform .3s ease;
}}
.logo:hover {{ transform:scale(1.08) rotate(-1deg); }}

/* ── Animation ── */
@keyframes slideUp {{
    from {{ opacity:0; transform:translateY(24px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}

/* ── Scheme card ── */
.scheme-card {{
    background: white;
    border-radius: var(--r-lg);
    padding: 26px 22px 20px;
    box-shadow: var(--sh-s);
    border: 1px solid rgba(64,145,108,.12);
    border-top: 4px solid var(--g-main);
    transition: transform .25s ease, box-shadow .25s ease, border-top-color .25s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 10px;
    position: relative;
    overflow: hidden;
}}
.scheme-card::before {{
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 110px; height: 110px;
    border-radius: 50%;
    background: var(--g-pale);
    opacity: .5;
    transition: transform .3s ease;
}}
.scheme-card:hover {{
    transform: translateY(-6px);
    box-shadow: var(--sh-c);
    border-top-color: var(--e-light);
}}
.scheme-card:hover::before {{ transform: scale(1.4); }}
.scheme-icon {{ font-size: 34px; line-height: 1; }}
.scheme-name {{
    font-family: var(--ff-d);
    font-size: 16px;
    font-weight: 700;
    color: var(--g-dark);
    line-height: 1.3;
}}
.scheme-desc {{
    font-size: 12.5px;
    color: #5a6e5a;
    line-height: 1.65;
    flex: 1;
}}
.scheme-link {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    font-weight: 700;
    color: var(--g-main);
    text-decoration: none;
    padding: 6px 14px;
    border-radius: 50px;
    background: var(--g-pale);
    border: 1px solid rgba(64,145,108,.2);
    transition: background .2s ease, color .2s ease;
    align-self: flex-start;
    margin-top: 4px;
}}
.scheme-link:hover {{
    background: var(--g-main);
    color: white;
    text-decoration: none;
}}
.scheme-tag {{
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: var(--e-mid);
    background: #fdf3e7;
    border-radius: 50px;
    padding: 3px 10px;
    border: 1px solid rgba(139,94,60,.15);
    width: fit-content;
}}

@media (max-width:768px) {{
    .hero-title {{ font-size:34px; }}
    .npk-grid   {{ grid-template-columns:1fr; }}
}}
</style>
{logo_html}
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error("⚠️ `crop_model.pkl` not found. Place it in the app directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ═══════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-badge">🇮🇳 Made for Indian Farmers</div>
    <div class="hero-title">Farm<span>Genius</span></div>
    <div class="hero-sub">AI-Powered Smart Crop Recommendation &amp; Soil Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 24px;'>
        <div style='font-size:36px;'>🧪</div>
        <div style='font-family:"Playfair Display",serif;font-size:22px;font-weight:700;color:#b7e4c7;'>Soil Lab</div>
        <div style='font-size:12px;opacity:.6;margin-top:4px;'>Enter your field measurements</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Nutrient Levels**")
    N = st.slider("🌿 Nitrogen (N)",   0, 200, 50, key="sl_N")
    P = st.slider("🪨 Phosphorus (P)", 0, 200, 50, key="sl_P")
    K = st.slider("⚗️ Potassium (K)",  0, 200, 50, key="sl_K")

    st.markdown("---")
    st.markdown("**Soil Properties**")
    ph       = st.slider("🔬 pH Level",      0.0, 14.0, 6.5, step=0.1, key="sl_ph")
    rainfall = st.slider("🌧 Rainfall (mm)", 0,   300,  100,           key="sl_rain")

    st.markdown("---")
    st.markdown("**Weather Fetch**")
    city = st.text_input("📍 Your City", placeholder="e.g. Jaipur", key="city_input")

    if st.button("🌦 Get Live Weather", key="weather_btn"):
        if not city.strip():
            st.warning("Please enter a city name.")
        else:
            with st.spinner("Fetching weather..."):
                temp, hum = fetch_weather(city)
                if temp is not None:
                    st.session_state.temp = temp
                    st.session_state.hum  = hum
                    st.success(f"✅ Weather loaded for {city.title()}!")

    st.markdown("---")
    if st.session_state.temp is not None:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,.10);border-radius:12px;
                    padding:14px 18px;text-align:center;'>
            <div style='font-size:28px;'>🌤</div>
            <div style='font-size:22px;font-weight:700;color:#b7e4c7;'>
                {st.session_state.temp}°C
            </div>
            <div style='font-size:13px;opacity:.7;'>Humidity: {st.session_state.hum}%</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# WEATHER CARDS (main area)
# ═══════════════════════════════════════════════

if st.session_state.temp is not None:
    w1, w2, w3, w4 = st.columns(4)
    for col, icon, label, value, bg in [
        (w1, "🌡", "Temperature", f"{st.session_state.temp}°C",
         "linear-gradient(135deg,#1b4332,#2d6a4f)"),
        (w2, "💧", "Humidity",    f"{st.session_state.hum}%",
         "linear-gradient(135deg,#1b4332,#40916c)"),
        (w3, "🌧", "Rainfall",    f"{rainfall} mm",
         "linear-gradient(135deg,#5c3d2e,#8b5e3c)"),
        (w4, "🔬", "Soil pH",     str(ph),
         "linear-gradient(135deg,#2d4a3e,#4a7c59)"),
    ]:
        with col:
            st.markdown(f"""
            <div class="w-card" style="background:{bg};">
                <div class="w-icon">{icon}</div>
                <div class="w-label">{label}</div>
                <div class="w-value">{value}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PREDICTION ENGINE
# ═══════════════════════════════════════════════

st.markdown("""
<div class="sec-head">🚀 Crop Prediction Engine</div>
<div class="sec-sub">Our ML model analyzes 7 soil &amp; weather parameters to find your best crop</div>
""", unsafe_allow_html=True)

if st.button("🌾 Predict Best Crop for My Field", key="predict_btn"):
    if st.session_state.temp is None:
        st.warning("⚠️ Please fetch live weather from the sidebar first.")
    else:
        with st.spinner("Analyzing soil & climate data..."):
            try:
                sample = np.array([[N, P, K,
                                    st.session_state.temp,
                                    st.session_state.hum,
                                    ph, rainfall]])
                predicted = model.predict(sample)[0].lower().strip()
                st.session_state.prediction = predicted
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.session_state.prediction = None

if st.session_state.prediction:
    pred_crop = st.session_state.prediction
    em        = CROP_EMOJI.get(pred_crop, "🌾")
    fn, fd, fc = fertilizer_advice(N, P, K)
    loc_name  = city.strip().title() if city.strip() else "your location"

    st.markdown(f"""
    <div class="result-card">
        <div class="result-lbl">Recommended Crop</div>
        <div style="font-size:56px;margin:8px 0;">{em}</div>
        <div class="result-name">{pred_crop}</div>
        <div style="font-size:14px;opacity:.7;margin-top:8px;">
            Based on soil nutrients &amp; {loc_name} weather data
        </div>
    </div>
    """, unsafe_allow_html=True)

    img_pred = CROP_IMAGES.get(pred_crop, "")
    if os.path.exists(img_pred):
        st.markdown("<br>", unsafe_allow_html=True)
        _, ci_mid, _ = st.columns([1, 2, 1])
        with ci_mid:
            st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
            safe_image(img_pred, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="fert-card" style="border-left-color:{fc};">
        <div class="fert-name" style="color:{fc};">{fn}</div>
        <div class="fert-desc">{fd}</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SOIL NUTRIENT ANALYSIS
# ═══════════════════════════════════════════════

st.markdown('<div class="divider"><span>🌿</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">📊 Soil Nutrient Analysis</div>
<div class="sec-sub">Visualize the NPK balance in your soil</div>
""", unsafe_allow_html=True)

total = N + P + K
if total == 0:
    st.info("Move the N, P, K sliders in the sidebar to see the breakdown.")
else:
    pct_n = (N / total) * 100
    pct_p = (P / total) * 100
    pct_k = (K / total) * 100

    na1, na2, na3 = st.columns([1, 1.4, 1])

    with na1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        for label, val, bg in [
            ("🌿 Nitrogen",   f"{N} mg/kg",   "#d8f3dc"),
            ("🪨 Phosphorus", f"{P} mg/kg",   "#f3e8d8"),
            ("⚗️ Potassium",  f"{K} mg/kg",   "#dceeff"),
            ("🔬 pH Level",   str(ph),          "#f0f0f0"),
            ("🌧 Rainfall",   f"{rainfall} mm", "#e8f0fe"),
        ]:
            st.markdown(
                f'<div class="pill" style="background:{bg};">'
                f'<b>{label}:</b> {val}</div>',
                unsafe_allow_html=True,
            )

    with na2:
        labels = ["Nitrogen", "Phosphorus", "Potassium"]
        values = [pct_n, pct_p, pct_k]
        colors = ["#2d6a4f", "#8b5e3c", "#40916c"]

        fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
        ax.pie(
            values, labels=None, colors=colors, autopct="%1.1f%%",
            startangle=140, pctdistance=0.78, explode=(0.04, 0.04, 0.04),
            wedgeprops=dict(linewidth=2.5, edgecolor="white"),
        )
        for txt in ax.texts:
            if "%" in txt.get_text():
                txt.set(color="white", fontsize=11, fontweight="bold")
        patches = [
            mpatches.Patch(color=colors[i], label=f"{labels[i]}: {values[i]:.1f}%")
            for i in range(3)
        ]
        ax.legend(handles=patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=10)
        ax.set_title("NPK Distribution", fontsize=13,
                     fontweight="bold", color="#1b4332", pad=14)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with na3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        for label, pct, grad in [
            ("Nitrogen",   pct_n, "linear-gradient(90deg,#74c69d,#2d6a4f)"),
            ("Phosphorus", pct_p, "linear-gradient(90deg,#d4a96a,#8b5e3c)"),
            ("Potassium",  pct_k, "linear-gradient(90deg,#74c69d,#1b4332)"),
        ]:
            st.markdown(f"""
            <div class="bar-wrap">
                <div class="bar-lbl"><span>{label}</span><span>{pct:.1f}%</span></div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{pct:.1f}%;background:{grad};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# CROP COMPARISON
# ═══════════════════════════════════════════════

st.markdown('<div class="divider"><span>⚖️</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">🔍 Crop Comparison Tool</div>
<div class="sec-sub">Select any two crops to instantly compare NPK needs and soil suitability</div>
""", unsafe_allow_html=True)

cm1, cm2 = st.columns(2)
with cm1:
    crop_a = st.selectbox(
        "Crop 1", CROP_LIST, index=0, key="cmp_a",
        format_func=lambda c: f"{CROP_EMOJI.get(c,'')} {c.capitalize()}"
    )
with cm2:
    crop_b = st.selectbox(
        "Crop 2", CROP_LIST, index=1, key="cmp_b",
        format_func=lambda c: f"{CROP_EMOJI.get(c,'')} {c.capitalize()}"
    )

s_a   = suitability_score(N, P, K, crop_a)
s_b   = suitability_score(N, P, K, crop_b)
req_a = NPK_REQ[crop_a]
req_b = NPK_REQ[crop_b]

st.markdown("<br>", unsafe_allow_html=True)
cc1, cc2, cc3 = st.columns([1, 1.2, 1])

with cc1:
    score_card(crop_a, s_a, s_b,
               "#40916c", "linear-gradient(90deg,#74c69d,#2d6a4f)")

with cc2:
    cats = ["N", "P", "K"]
    x    = np.arange(3)
    bw   = 0.32
    fig3, ax3 = plt.subplots(figsize=(4, 3.4), facecolor="none")
    ax3.bar(x - bw/2, [req_a["N"], req_a["P"], req_a["K"]], bw,
            color="#40916c", alpha=.9, label=crop_a.capitalize(),
            edgecolor="white", linewidth=1.5, zorder=3)
    ax3.bar(x + bw/2, [req_b["N"], req_b["P"], req_b["K"]], bw,
            color="#8b5e3c", alpha=.9, label=crop_b.capitalize(),
            edgecolor="white", linewidth=1.5, zorder=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(cats, fontsize=12, fontweight="bold", color="#1b4332")
    ax3.set_ylabel("mg/kg", fontsize=9, color="#888")
    ax3.set_title("NPK Requirements", fontsize=11,
                  fontweight="bold", color="#1b4332", pad=12)
    ax3.legend(fontsize=9, frameon=False, loc="upper right")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.spines[["left", "bottom"]].set_color("#e0e0e0")
    ax3.tick_params(colors="#888", length=0)
    ax3.set_facecolor("none")
    ax3.yaxis.grid(True, color="#f0f0f0", zorder=0)
    fig3.patch.set_alpha(0)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    if s_a == s_b:
        verdict, vc = "Both crops are equally suitable!", "#555"
    elif s_a > s_b:
        verdict, vc = f"🌾 {crop_a.capitalize()} suits your soil better", "#40916c"
    else:
        verdict, vc = f"🌾 {crop_b.capitalize()} suits your soil better", "#40916c"
    st.markdown(f"""
    <div style="text-align:center;margin-top:10px;padding:10px;
                background:#f0faf4;border-radius:10px;
                font-weight:600;font-size:13px;color:{vc};">
        {verdict}
    </div>""", unsafe_allow_html=True)

with cc3:
    score_card(crop_b, s_b, s_a,
               "#8b5e3c", "linear-gradient(90deg,#d4a96a,#8b5e3c)")

# ═══════════════════════════════════════════════
# CROP KNOWLEDGE BASE
# ═══════════════════════════════════════════════

st.markdown('<div class="divider"><span>🌾</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">🌾 Crop Knowledge Base</div>
<div class="sec-sub">Explore ideal growing conditions, market price, and soil needs for each crop</div>
""", unsafe_allow_html=True)

selected = st.selectbox(
    "Choose a crop to explore",
    CROP_LIST,
    format_func=lambda c: f"{CROP_EMOJI.get(c,'🌿')} {c.capitalize()}",
    key="explorer",
)

crop_k   = selected.lower()
msp_val  = MSP.get(crop_k, 0)
req_k    = NPK_REQ.get(crop_k, {})
desc_k   = CROP_DESC.get(crop_k, "")
pl_label = "Govt. MSP Price" if crop_k in GOVT_MSP_CROPS else "Est. Market Price"

ex1, ex2 = st.columns([1, 1.6])

with ex1:
    img_k = CROP_IMAGES.get(crop_k, "")
    st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
    if not safe_image(img_k, use_container_width=True):
        st.markdown(f"""
        <div style="background:var(--g-pale);border-radius:14px;height:220px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:72px;">
            {CROP_EMOJI.get(crop_k,'🌿')}
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ex2:
    st.markdown(f"""
    <div class="crop-card">
        <div style="font-size:36px;margin-bottom:4px;">{CROP_EMOJI.get(crop_k,'🌾')}</div>
        <div class="crop-card-name">{crop_k.capitalize()}</div>
        <div class="price-badge">₹{msp_val:,} / quintal &nbsp;·&nbsp; {pl_label}</div>
        <div class="npk-grid">
            <div class="npk-box">
                <div class="npk-label">🌿 Nitrogen</div>
                <div class="npk-val">{req_k.get('N', '-')}</div>
            </div>
            <div class="npk-box">
                <div class="npk-label">🪨 Phosphorus</div>
                <div class="npk-val">{req_k.get('P', '-')}</div>
            </div>
            <div class="npk-box">
                <div class="npk-label">⚗️ Potassium</div>
                <div class="npk-val">{req_k.get('K', '-')}</div>
            </div>
        </div>
        <div class="crop-desc">📌 {desc_k}</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# GOVERNMENT FARMING SCHEMES
# ═══════════════════════════════════════════════

st.markdown('<div class="divider"><span>🏛️</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">🏛️ Government Farming Schemes</div>
<div class="sec-sub">Central government programmes designed to support and protect Indian farmers</div>
""", unsafe_allow_html=True)

sch_cols = st.columns(5)
for i, scheme in enumerate(SCHEMES):
    with sch_cols[i]:
        st.markdown(f"""
        <div class="scheme-card">
            <div class="scheme-icon">{scheme['icon']}</div>
            <div class="scheme-tag">{scheme['tag']}</div>
            <div class="scheme-name">{scheme['name']}<br>
                <span style="font-family:var(--ff-b);font-size:10.5px;
                             color:#888;font-weight:400;">{scheme['full']}</span>
            </div>
            <div class="scheme-desc">{scheme['desc']}</div>
            <a class="scheme-link" href="{scheme['link']}" target="_blank" rel="noopener">
                🔗 Learn More
            </a>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SMART FARMING TIPS
# ═══════════════════════════════════════════════

st.markdown('<div class="divider"><span>💡</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">💡 Smart Farming Tips</div>
<div class="sec-sub">Simple practices to maximize your yield and soil health</div>
""", unsafe_allow_html=True)

t_cols = st.columns(5)
for i, (icon, head, text) in enumerate(TIPS):
    with t_cols[i]:
        st.markdown(f"""
        <div class="tip">
            <div class="tip-icon">{icon}</div>
            <div class="tip-head">{head}</div>
            <div class="tip-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:28px 0 16px;
            border-top:1px solid rgba(64,145,108,.15);">
    <div style="font-family:'Playfair Display',serif;font-size:20px;
                font-weight:700;color:#2d6a4f;">🌾 FarmGenius</div>
    <div style="font-size:12px;color:#9aab9a;margin-top:6px;">
        Empowering Indian Farmers with AI &nbsp;·&nbsp;
        Built with ❤️ for Bharat's Kisaan
    </div>
</div>
""", unsafe_allow_html=True)