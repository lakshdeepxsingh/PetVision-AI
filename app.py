import streamlit as st
import numpy as np
import time
import base64
from datetime import datetime
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PetVision AI — Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

CLASSIFICATION_THRESHOLD = 0.5

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "model/cat_dog_model.keras"
MODEL_NAME = "DeepVision Net"
MODEL_VERSION = "v2.4.1"


@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)


def get_model():
    try:
        return load_my_model(), None
    except Exception as e:
        return None, str(e)


# ============================================================
# NAV ICONS
# ============================================================

ICON_SVGS = {
    "home": "<path d='M3 9.5 12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V9.5Z'/>",
    "bar-chart": "<path d='M4 20V10'/><path d='M12 20V4'/><path d='M20 20v-7'/>",
    "clock": "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 3'/>",
    "bookmark": "<path d='M7 3h10a1 1 0 0 1 1 1v17l-6-4-6 4V4a1 1 0 0 1 1-1Z'/>",
}


def icon_css_rule(nth, icon_key, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{ICON_SVGS[icon_key]}</svg>"
    svg = svg.replace("#", "%23").replace('"', "'")
    return (
        f'[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type({nth})::before '
        f'{{ background-image: url("data:image/svg+xml,{svg}"); }}'
    )


NAV_ICON_KEYS = ["home", "bar-chart", "clock", "bookmark"]
_icon_rules = []
for _i, _key in enumerate(NAV_ICON_KEYS, start=1):
    _icon_rules.append(icon_css_rule(_i, _key, "%23b8b0db"))
    _active_rule = icon_css_rule(_i, _key, "%23ffffff").replace(
        "::before {", ":has(input:checked)::before {"
    )
    _icon_rules.append(_active_rule)
ICON_CSS = "\n".join(_icon_rules)

# ============================================================
# STYLES (EXACT MATCH FROSTED GLASS SIDEBAR + DASHBOARD)
# ============================================================

BASE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

#MainMenu, footer, header {{ visibility: hidden; display: none; }}

.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1.5rem !important;
}}

/* ---- Dark App Background with Subtle Violet Glow ---- */
.stApp {{
    background: radial-gradient(circle at 10% 20%, #20113b 0%, #0c071a 50%, #05030d 100%) !important;
    background-attachment: fixed !important;
}}

/* ---- Frosted Glass Sidebar (Translucent & Glowing Border) ---- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(88, 28, 135, 0.25) 0%, rgba(20, 10, 38, 0.55) 100%) !important;
    backdrop-filter: blur(35px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(35px) saturate(200%) !important;
    border-right: 1.5px solid rgba(255, 255, 255, 0.25) !important;
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5) !important;
    width: 340px !important;
    min-width: 340px !important;
}}

[data-testid="stSidebarHeader"] {{ display: none !important; }}

[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
    margin-top: 0 !important;
    width: 340px !important;
}}

[data-testid="stSidebarUserContent"] {{ padding: 0 !important; margin: 0 !important; }}
[data-testid="stSidebar"] section {{ padding: 0 !important; }}

.pv-logo {{
    display: flex; align-items: center; gap: 10px;
    padding: 1rem 1.1rem 0.8rem 1.1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    margin-bottom: 0.7rem;
}}
.pv-logo-icon {{
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg,#c084fc,#7e22ce);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
    box-shadow: 0 4px 15px rgba(192, 132, 252, 0.4);
}}
.pv-logo-text-main {{ color: #ffffff; font-family: 'Space Grotesk',sans-serif; font-weight:700; font-size: 17px; }}
.pv-logo-text-sub {{ color: #e9d5ff; font-size: 12px; }}

.pv-section-label {{
    color: #e9d5ff; font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    padding: 0 1.1rem; margin: 1.1rem 0 .5rem 0; text-transform: uppercase;
}}

/* Dropzone Glass */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {{ padding: 0 1.1rem !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1.5px dashed rgba(255, 255, 255, 0.3) !important;
    border-radius: 18px !important;
    padding: 1.2rem 0.8rem !important;
    backdrop-filter: blur(16px) !important;
}}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {{
    display: flex !important; flex-direction: column !important; align-items: center !important; gap: 8px !important;
}}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
    margin: 0 auto !important; border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.2) !important; color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
}}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p {{
    color: #ffffff !important; text-align: center !important; font-size: 12px !important;
}}

/* Navigation Radio Glass Buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] {{ padding: 0 0.5rem !important; margin: 0 !important; width: 100% !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{ width: 100% !important; }}

[data-testid="stSidebar"] div[role="radiogroup"] {{
    display: flex !important; flex-direction: column !important; align-items: stretch !important;
    gap: 6px !important; padding: 0 !important; width: 100% !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label {{
    display: flex !important; align-items: center !important; width: 100% !important;
    box-sizing: border-box !important; border-radius: 14px !important;
    padding: 12px 1.2rem !important; margin: 0 !important; cursor: pointer;
    transition: all .2s ease;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label::before {{
    content: ""; width: 20px; height: 20px; margin-right: 12px; flex-shrink: 0;
    background-repeat: no-repeat; background-position: center; background-size: contain;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(255, 255, 255, 0.1); }}

/* Active Selection Glass Tab (Matching Reference Gray-White Glow) */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: rgba(255, 255, 255, 0.22) !important;
    backdrop-filter: blur(15px) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.45) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
}}

/* Active Selection Indicator Dot (Red/Orange Accent like reference) */
[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"]:checked + div,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[aria-hidden="true"],
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[role="radio"] {{
    background-color: #ff5252 !important;
    border-color: #ff5252 !important;
    box-shadow: 0 0 8px #ff5252 !important;
}}

[data-testid="stSidebar"] div[role="radiogroup"] label p {{ color: #e9d5ff !important; font-size: 15px; font-weight: 500; margin: 0; }}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color: #ffffff !important; font-weight: 600; }}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display:none; }}

{ICON_CSS}

/* ---- OUTER DASHBOARD GLASS SHELL (Holds Photo Backdrop) ---- */
div[class*="st-key-outer_shell"] {{
    background-color: rgba(30, 20, 55, 0.7) !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    border: 1.5px solid rgba(255, 255, 255, 0.35) !important;
    border-radius: 32px !important;
    padding: 28px !important;
    box-shadow: 
        0 30px 70px rgba(0, 0, 0, 0.6),
        inset 0 1.5px 2px rgba(255, 255, 255, 0.5) !important;
    position: relative !important;
    overflow: hidden !important;
}}

/* Titles */
.pv-title {{ 
    font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:34px; color:#ffffff; margin-bottom:4px; 
    text-shadow: 0 2px 10px rgba(0,0,0,0.6); 
}}
.pv-subtitle {{ color:#f3e8ff; font-size:15px; margin-bottom:1.5rem; text-shadow: 0 1px 4px rgba(0,0,0,0.5); }}

/* ---- INNER FROSTED GLASS PANELS ---- */
div[class*="st-key-panel_"] {{
    background: rgba(255, 255, 255, 0.16) !important;
    backdrop-filter: blur(25px) saturate(190%) !important;
    -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.45) !important;
    border-radius: 24px !important;
    padding: 22px !important;
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.4),
        inset 0 1.5px 1.5px rgba(255, 255, 255, 0.6) !important;
    margin-bottom: 20px !important;
}}

.pv-panel-title {{
    color:#ffffff; font-size:12.5px; font-weight:700; letter-spacing:1px; text-transform:uppercase;
    margin-bottom:14px; display:flex; align-items:center; gap:8px; text-shadow: 0 1px 4px rgba(0,0,0,0.5);
}}

/* Preview Image Glass Frame */
div[class*="st-key-panel_img"] [data-testid="stImage"] img {{
    max-width: 250px !important;
    max-height: 250px !important;
    border-radius: 20px !important; 
    border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5) !important;
    display: block; 
    margin: 0 auto;
}}

.pv-label-tag {{
    display:flex; align-items:center; gap:8px; font-family:'Space Grotesk',sans-serif;
    font-weight:700; font-size:42px; color:#ffffff; text-shadow: 0 4px 15px rgba(0,0,0,0.5);
}}
.pv-conf-num {{
    font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:42px;
    background: linear-gradient(90deg,#ffffff,#f3e8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.5));
}}
.pv-eyebrow {{ color:#f3e8ff; font-size:12px; font-weight:600; letter-spacing:1px; margin-bottom:4px; text-shadow: 0 1px 3px rgba(0,0,0,0.5); }}
.pv-badge {{
    display:inline-block; margin-top:10px; padding:6px 18px; border-radius:999px;
    background: rgba(255, 255, 255, 0.25); color: #ffffff; font-size: 12.5px; font-weight: 600; 
    border: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3); backdrop-filter: blur(12px);
}}
.pv-caption {{ color:#f3e8ff; font-size:13.5px; margin-top:14px; text-shadow: 0 1px 3px rgba(0,0,0,0.5); }}
.pv-caption b {{ color:#ffffff; }}

.pv-progress-row {{ display:flex; align-items:center; gap:14px; }}
.pv-progress-track {{ 
    flex:1; background: rgba(0, 0, 0, 0.35); border-radius:999px; height:12px; overflow:hidden; border: 1px solid rgba(255, 255, 255, 0.25); 
}}
.pv-progress-fill {{ 
    background: linear-gradient(90deg, #c084fc, #f472b6); height:100%; border-radius:999px; box-shadow: 0 0 15px rgba(244, 114, 182, 0.8);
}}
.pv-progress-pct {{ color:#ffffff; font-weight:700; font-size:14px; width:44px; text-align:right; }}
.pv-progress-meta {{ display:flex; justify-content:space-between; color:#f3e8ff; font-size:12.5px; margin-top:8px; }}

.pv-empty {{ color:#f3e8ff; font-size:14px; text-align:center; padding:2.4rem 0; }}
</style>
"""

# Render base styling
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="pv-logo">
            <div class="pv-logo-icon">🐾</div>
            <div>
                <div class="pv-logo-text-main">PetVision AI</div>
                <div class="pv-logo-text-sub">Image Classifier</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pv-section-label">Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop an image, or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="pv-section-label">Navigation</div>', unsafe_allow_html=True)
    nav_labels = ["Dashboard", "Predictions", "History", "Saved Results"]
    current_idx = nav_labels.index(st.session_state.page)
    selected = st.radio(
        "nav", nav_labels, index=current_idx, label_visibility="collapsed", key="nav_radio"
    )
    st.session_state.page = selected

# ============================================================
# APPLY PHOTO BACKDROP ONLY INSIDE OUTER DASHBOARD SHELL
# ============================================================

if uploaded_file is not None:
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    b64_str = base64.b64encode(img_bytes).decode()
    
    st.markdown(
        f"""
        <style>
        div[class*="st-key-outer_shell"] {{
            background: linear-gradient(rgba(15, 8, 35, 0.45), rgba(15, 8, 35, 0.55)), 
                        url("data:image/jpeg;base64,{b64_str}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file.seek(0)

model, model_error = get_model()

# ============================================================
# PREDICT HELPER
# ============================================================


def run_prediction(pil_img: Image.Image, filename: str):
    img_resized = pil_img.convert("RGB").resize((150, 150))
    arr = keras_image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0) / 255.0

    start = time.time()
    raw = model.predict(arr, verbose=0)[0][0]
    elapsed = time.time() - start

    is_dog = raw > CLASSIFICATION_THRESHOLD
    confidence = (raw if is_dog else (1 - raw)) * 100
    label = "Dog" if is_dog else "Cat"

    if confidence >= 90:
        conf_tag = "Very High Confidence"
    elif confidence >= 70:
        conf_tag = "High Confidence"
    elif confidence >= 55:
        conf_tag = "Moderate Confidence"
    else:
        conf_tag = "Low Confidence"

    return {
        "filename": filename,
        "label": label,
        "confidence": round(float(confidence), 2),
        "conf_tag": conf_tag,
        "elapsed": round(elapsed, 2),
        "timestamp": datetime.now().strftime("%b %d, %Y · %I:%M %p"),
        "saved": False,
        "thumbnail": pil_img,
    }


# ============================================================
# DASHBOARD PAGE
# ============================================================


def render_dashboard():
    with st.container(key="outer_shell"):
        st.markdown('<div class="pv-title">Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pv-subtitle">Upload an image and let the AI predict whether it is a Cat or a Dog.</div>',
            unsafe_allow_html=True,
        )

        if model_error:
            st.error(f"Couldn't load the model at `{MODEL_PATH}`: {model_error}")
            return

        # ---- Image preview ----
        with st.container(key="panel_img"):
            st.markdown('<div class="pv-panel-title">🖼️ Image Preview</div>', unsafe_allow_html=True)
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file)
                st.image(pil_img, use_container_width=False)
            else:
                st.markdown(
                    '<div class="pv-empty">Upload an image from the sidebar to see it here.</div>',
                    unsafe_allow_html=True,
                )
                pil_img = None

        if uploaded_file is None or pil_img is None:
            return

        record = run_prediction(pil_img, uploaded_file.name)
        already_logged = (
            st.session_state.history
            and st.session_state.history[-1]["filename"] == record["filename"]
            and st.session_state.history[-1]["confidence"] == record["confidence"]
        )
        if not already_logged:
            st.session_state.history.append(record)

        # ---- Prediction result ----
        with st.container(key="panel_prediction"):
            st.markdown('<div class="pv-panel-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
            icon = "🐶" if record["label"] == "Dog" else "🐱"
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown('<div class="pv-eyebrow">PREDICTED LABEL</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="pv-label-tag">{record["label"]} <span>{icon}</span></div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="pv-eyebrow">CONFIDENCE SCORE</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="pv-conf-num">{record["confidence"]:.2f}%</div>', unsafe_allow_html=True)

            st.markdown(f'<span class="pv-badge">⭐ {record["conf_tag"]}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="pv-caption">The model predicts this image contains a <b>{record["label"]}</b>.</div>',
                unsafe_allow_html=True,
            )
            if st.button("⭐ Save this result", key="save_dashboard"):
                st.session_state.history[-1]["saved"] = True
                st.toast("Saved to your results.")

        # ---- Processing bar ----
        with st.container(key="panel_processing"):
            st.markdown('<div class="pv-panel-title">⚡ Processing Image</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="pv-progress-row">
                    <div class="pv-progress-track"><div class="pv-progress-fill" style="width:100%;"></div></div>
                    <div class="pv-progress-pct">100%</div>
                </div>
                <div class="pv-progress-meta">
                    <span>✅ Analysis complete</span>
                    <span>Processed in {record['elapsed']}s</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PREDICTIONS / HISTORY / SAVED PAGE
# ============================================================


def render_list_page(key_prefix, title, subtitle, records, empty_msg):
    with st.container(key="outer_shell"):
        st.markdown(f'<div class="pv-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pv-subtitle">{subtitle}</div>', unsafe_allow_html=True)

        with st.container(key=f"panel_{key_prefix}_list"):
            if not records:
                st.markdown(f'<div class="pv-empty">{empty_msg}</div>', unsafe_allow_html=True)
                return

            for i, rec in enumerate(reversed(records)):
                real_idx = len(records) - 1 - i
                icon = "🐶" if rec["label"] == "Dog" else "🐱"
                cols = st.columns([0.7, 2.3, 1.4, 1.6, 1])
                with cols[0]:
                    st.image(rec["thumbnail"], width=56)
                with cols[1]:
                    st.markdown(
                        f"**{rec['filename']}**  \n<span style='color:#e9d5ff;font-size:12.5px;'>{rec['timestamp']}</span>",
                        unsafe_allow_html=True,
                    )
                with cols[2]:
                    st.markdown(f"**{icon} {rec['label']}**")
                with cols[3]:
                    st.markdown(
                        f"<span style='color:#f3e8ff;font-weight:700;'>{rec['confidence']:.2f}%</span> "
                        f"<span style='color:#e9d5ff;font-size:12px;'>· {rec['conf_tag']}</span>",
                        unsafe_allow_html=True,
                    )
                with cols[4]:
                    if rec["saved"]:
                        st.markdown("⭐ Saved")
                    else:
                        if st.button("Save", key=f"save_{key_prefix}_{real_idx}"):
                            st.session_state.history[real_idx]["saved"] = True
                            st.rerun()
                st.markdown("<hr style='border-color:rgba(255, 255, 255, 0.25); margin:6px 0;'>", unsafe_allow_html=True)


# ============================================================
# ROUTER
# ============================================================

page = st.session_state.page
if page == "Dashboard":
    render_dashboard()
elif page == "Predictions":
    render_list_page(
        "predictions", "Predictions", "Every image you've classified this session.",
        st.session_state.history, "No predictions yet — upload an image from the Dashboard.",
    )
elif page == "History":
    render_list_page(
        "history", "History", "A chronological log of your classification activity.",
        st.session_state.history, "Nothing here yet.",
    )
elif page == "Saved Results":
    saved = [r for r in st.session_state.history if r["saved"]]
    render_list_page(
        "saved", "Saved Results", "Results you've bookmarked for later.",
        saved, "You haven't saved any results yet.",
    )