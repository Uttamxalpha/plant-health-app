"""
╔══════════════════════════════════════════════════════════╗
║       PLANT HEALTH CHECKING SYSTEM — STREAMLIT APP       ║
╚══════════════════════════════════════════════════════════╝
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import json
import random
import io
import base64

# ── Page Config (MUST be first Streamlit call) ──────────────
st.set_page_config(
    page_title="Plant Health AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Space Mono', monospace !important; }

  .stApp {
    background: #060a0f;
    color: #c8d8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1520 !important;
    border-right: 1px solid #1e3050 !important;
  }
  [data-testid="stSidebar"] * { color: #c8d8f0 !important; }

  /* ── Header Banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #0d1520 0%, #0a1f14 50%, #0d1520 100%);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse 40% 40% at 70% 50%, rgba(0,230,118,0.06) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem;
    font-weight: 800;
    color: #e8f4ff;
    margin: 0 0 8px 0;
    line-height: 1.1;
  }
  .hero-title span { color: #00e676; }
  .hero-sub { font-size: 13px; color: #546e8a; letter-spacing: 1px; margin: 0; }
  .hero-badge {
    display: inline-block;
    background: rgba(0,230,118,0.12);
    border: 1px solid rgba(0,230,118,0.3);
    color: #00e676;
    font-size: 10px;
    letter-spacing: 3px;
    padding: 5px 14px;
    border-radius: 2px;
    margin-bottom: 16px;
    text-transform: uppercase;
  }

  /* ── Cards ── */
  .metric-card {
    background: #0d1520;
    border: 1px solid #1e3050;
    border-radius: 8px;
    padding: 20px 16px;
    text-align: center;
  }
  .metric-num {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem;
    font-weight: 800;
    color: #e8f4ff;
    display: block;
  }
  .metric-label { font-size: 10px; color: #546e8a; letter-spacing: 2px; text-transform: uppercase; }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
    background: #0d1520 !important;
    border: 2px dashed #1e3050 !important;
    border-radius: 10px !important;
    padding: 20px !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #00e676 !important;
  }

  /* ── Report cards ── */
  .report-card {
    background: #0d1520;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .report-healthy {
    border-left: 4px solid #00e676;
  }
  .report-diseased {
    border-left: 4px solid #ff1744;
  }
  .report-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.2rem;
    font-weight: 800;
    color: #e8f4ff;
    margin-bottom: 4px;
  }
  .tag-healthy {
    display: inline-block;
    background: rgba(0,230,118,0.12);
    border: 1px solid rgba(0,230,118,0.3);
    color: #00e676;
    font-size: 11px;
    padding: 3px 12px;
    border-radius: 4px;
    font-weight: 700;
  }
  .tag-diseased {
    display: inline-block;
    background: rgba(255,23,68,0.12);
    border: 1px solid rgba(255,23,68,0.3);
    color: #ff1744;
    font-size: 11px;
    padding: 3px 12px;
    border-radius: 4px;
    font-weight: 700;
  }
  .info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1e3050;
    font-size: 13px;
  }
  .info-row:last-child { border-bottom: none; }
  .info-label { color: #546e8a; }
  .info-value { color: #e8f4ff; font-weight: 700; }

  /* ── Confidence bar ── */
  .conf-bar-wrap {
    background: #1e3050;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-top: 6px;
  }
  .conf-bar-fill-green { background: #00e676; height: 100%; border-radius: 4px; }
  .conf-bar-fill-blue  { background: #2979ff; height: 100%; border-radius: 4px; }
  .conf-bar-fill-red   { background: #ff1744; height: 100%; border-radius: 4px; }

  /* ── Prediction list ── */
  .pred-item {
    background: #121d2e;
    border: 1px solid #1e3050;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .pred-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px; }
  .pred-name { color: #c8d8f0; }
  .pred-pct  { color: #00e676; font-weight: 700; }

  /* ── Severity badges ── */
  .severity-healthy   { color: #00e676; }
  .severity-mild      { color: #ffea00; }
  .severity-moderate  { color: #ff9100; }
  .severity-severe    { color: #ff1744; }
  .severity-critical  { color: #d500f9; }

  /* ── Rec box ── */
  .rec-box {
    background: rgba(41,121,255,0.08);
    border: 1px solid rgba(41,121,255,0.25);
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 12px;
    font-size: 13px;
    color: #82b1ff;
    line-height: 1.7;
  }

  /* ── Section title ── */
  .section-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px;
    font-weight: 700;
    color: #546e8a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 16px;
    padding-left: 12px;
    border-left: 3px solid #00e676;
  }

  /* ── Demo mode notice ── */
  .demo-notice {
    background: rgba(255,234,0,0.06);
    border: 1px solid rgba(255,234,0,0.25);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 12px;
    color: #ffea00;
    margin-bottom: 16px;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #00e676 !important;
    color: #060a0f !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: #69f0ae !important;
    transform: translateY(-1px) !important;
  }

  /* ── Divider ── */
  hr { border-color: #1e3050 !important; }

  /* ── Progress bar ── */
  .stProgress > div > div { background: #00e676 !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #060a0f; }
  ::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 3px; }

  /* ── Hide default Streamlit elements ── */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

def _load_classes() -> list:
    """
    Prefer class_names.json generated by train.py so the Streamlit app
    always stays in sync with however the model was actually trained.
    Falls back to the hardcoded 38-class list.
    """
    for candidate in [
        "saved_models/class_names.json",
        "class_names.json",
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                classes = json.load(f)
            # st.sidebar shows this quietly
            return classes

    # Hardcoded fallback (PlantVillage 38-class standard)
    return [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew",
        "Cherry___healthy", "Corn___Cercospora_leaf_spot", "Corn___Common_rust",
        "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight", "Grape___healthy",
        "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper___Bacterial_spot", "Pepper___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy",
        "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
        "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
        "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]

CLASSES = _load_classes()

TREATMENT_MAP = {
    "healthy":            ("Healthy 🌿",    "No treatment needed. Keep up the good care!",              "severity-healthy"),
    "Apple_scab":         ("Mild–Moderate ⚠️", "Apply fungicide (captan/mancozeb). Prune infected leaves.", "severity-mild"),
    "Black_rot":          ("Moderate ⚠️",   "Remove infected parts. Apply copper-based fungicide.",       "severity-moderate"),
    "Late_blight":        ("Severe 🔴",     "Apply chlorothalonil immediately. Remove infected plants.",  "severity-severe"),
    "Early_blight":       ("Mild–Moderate ⚠️", "Improve air circulation. Apply azoxystrobin fungicide.", "severity-mild"),
    "Leaf_Mold":          ("Mild ⚠️",       "Reduce humidity. Apply copper hydroxide fungicide.",         "severity-mild"),
    "Spider_mites":       ("Moderate ⚠️",   "Apply miticide/neem oil. Increase plant humidity.",          "severity-moderate"),
    "Bacterial_spot":     ("Moderate ⚠️",   "Apply copper bactericide. Avoid overhead watering.",         "severity-moderate"),
    "Powdery_mildew":     ("Mild–Moderate ⚠️", "Apply sulfur-based fungicide. Improve air circulation.", "severity-mild"),
    "Mosaic_virus":       ("Severe 🔴",     "No cure. Remove infected plants. Control aphid vectors.",    "severity-severe"),
    "Yellow_Leaf_Curl":   ("Severe 🔴",     "No cure. Remove plants. Control whitefly vectors.",          "severity-severe"),
    "Haunglongbing":      ("Critical 🚨",   "No cure. Remove tree immediately. Control psyllid insects.", "severity-critical"),
    "Cedar_apple_rust":   ("Moderate ⚠️",   "Apply myclobutanil fungicide. Remove cedar galls nearby.",  "severity-moderate"),
    "Cercospora":         ("Moderate ⚠️",   "Apply fungicides containing azoxystrobin or pyraclostrobin.", "severity-moderate"),
    "Common_rust":        ("Moderate ⚠️",   "Apply fungicide at first sign. Plant resistant varieties.",  "severity-moderate"),
    "Northern_Leaf":      ("Moderate ⚠️",   "Apply propiconazole fungicide. Rotate crops annually.",      "severity-moderate"),
    "Esca":               ("Severe 🔴",     "Prune infected wood. Apply wound sealants. No full cure.",   "severity-severe"),
    "Leaf_blight":        ("Moderate ⚠️",   "Apply copper-based fungicide. Remove affected leaves.",      "severity-moderate"),
    "Leaf_scorch":        ("Moderate ⚠️",   "Improve drainage. Apply copper fungicide if bacterial.",     "severity-moderate"),
    "Septoria":           ("Moderate ⚠️",   "Remove infected leaves. Apply chlorothalonil fungicide.",    "severity-moderate"),
    "Target_Spot":        ("Moderate ⚠️",   "Apply azoxystrobin fungicide. Improve plant spacing.",       "severity-moderate"),
    "default":            ("Unknown ❓",    "Consult a plant pathologist for accurate diagnosis.",         "severity-mild"),
}

IMG_SIZE = (224, 224)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Try to load saved model; return None if not found (demo mode)."""
    model_paths = [
        "saved_models/plant_health_final.keras",
        "saved_models/best_model.keras",
        "plant_health_final.keras",
    ]
    try:
        import tensorflow as tf
        for path in model_paths:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path)
                return model, True
        return None, False
    except ImportError:
        return None, False
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None, False


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)


def real_predict(model, image: Image.Image, top_k=5):
    arr  = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:top_k]
    return [
        {"class": CLASSES[i], "confidence": float(preds[i]), "rank": r + 1}
        for r, i in enumerate(top_idx)
    ]


def demo_predict(image: Image.Image, plant_hint: str = "auto", top_k=5):
    """
    Simulated prediction when no trained model is available.
    Generates realistic-looking results based on optional plant type hint.
    """
    time.sleep(0.8)  # Simulate inference time

    # Filter classes by hint
    if plant_hint != "auto":
        pool = [c for c in CLASSES if c.lower().startswith(plant_hint.lower())]
        if not pool:
            pool = CLASSES
    else:
        pool = CLASSES

    # Pick a primary class
    primary = random.choice(pool)
    primary_conf = random.uniform(0.72, 0.97)

    # Build remaining probabilities
    others = [c for c in CLASSES if c != primary]
    random.shuffle(others)
    remaining = 1.0 - primary_conf
    top_results = [{"class": primary, "confidence": primary_conf, "rank": 1}]
    for r, cls in enumerate(others[:top_k - 1], 2):
        conf = remaining * random.uniform(0.05, 0.5)
        remaining -= conf
        top_results.append({"class": cls, "confidence": max(conf, 0.001), "rank": r})
    top_results.sort(key=lambda x: -x["confidence"])
    for r, p in enumerate(top_results, 1):
        p["rank"] = r
    return top_results


def build_report(predictions):
    """Build a structured health report from top predictions."""
    primary = predictions[0]
    cls     = primary["class"]
    parts   = cls.split("___")
    plant   = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in cls.lower()

    # Find treatment
    key = "healthy" if is_healthy else next(
        (k for k in TREATMENT_MAP if k.lower() in cls.lower().replace(" ", "_")),
        "default"
    )
    severity, recommendation, sev_cls = TREATMENT_MAP[key]

    return {
        "status":         "HEALTHY" if is_healthy else "DISEASED",
        "plant":          plant,
        "disease":        disease,
        "confidence":     primary["confidence"],
        "severity":       severity,
        "severity_class": sev_cls,
        "recommendation": recommendation,
        "predictions":    predictions,
        "is_healthy":     is_healthy,
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌿 Plant Health AI")
    st.markdown("---")

    st.markdown("**Settings**")
    plant_filter = st.selectbox(
        "Plant Type",
        ["auto", "Apple", "Tomato", "Corn", "Grape", "Potato",
         "Pepper", "Cherry", "Peach", "Strawberry", "Blueberry",
         "Raspberry", "Soybean", "Squash", "Orange"],
        help="Optionally narrow predictions to a specific plant type."
    )
    top_k = st.slider("Top predictions to show", 3, 10, 5)
    show_gradcam = st.checkbox("Show attention heatmap", value=True,
                               help="Visualize which leaf regions drove the prediction.")

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
<div style='font-size:11px; color:#546e8a; line-height:1.8'>
CNN with:<br>
• Residual connections<br>
• Channel attention (SE)<br>
• Depthwise separable conv<br>
• Grad-CAM explainability<br>
• 38 disease classes<br>
• 54K+ training images
</div>
""", unsafe_allow_html=True)

    # Show training summary if available
    summary_path = "saved_models/training_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            sm = json.load(f)
        st.markdown("**Last Training Run**")
        st.markdown(f"""
<div style='font-size:11px; color:#546e8a; line-height:2'>
  🎯 Val Accuracy : <span style='color:#00e676'>{sm.get('best_val_acc', 0)*100:.2f}%</span><br>
  🏆 Top-3 Acc   : <span style='color:#00e676'>{sm.get('best_top3', 0)*100:.2f}%</span><br>
  📈 AUC          : <span style='color:#00e676'>{sm.get('best_auc', 0):.4f}</span><br>
  🔄 Epochs run  : <span style='color:#c8d8f0'>{sm.get('epochs_run', '—')}</span><br>
  🏷 Classes     : <span style='color:#c8d8f0'>{sm.get('num_classes', '—')}</span>
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("""
<div style='font-size:10px; color:#1e3050; letter-spacing:1px'>
TENSORFLOW · KERAS · STREAMLIT
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">AI-Powered · CNN · TensorFlow</div>
  <h1 class="hero-title">Plant Health<br><span>Checking System</span></h1>
  <p class="hero-sub">Upload a leaf photo → Get instant disease diagnosis & treatment advice</p>
</div>
""", unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, "38", "Disease Classes"),
    (c2, "54K+", "Training Images"),
    (c3, "224²", "Input Resolution"),
    (c4, "~95%", "Accuracy"),
]:
    col.markdown(f"""
<div class="metric-card">
  <span class="metric-num">{num}</span>
  <span class="metric-label">{label}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load model
model, model_loaded = load_model()

# Demo notice
if not model_loaded:
    st.markdown("""
<div class="demo-notice">
  ⚡ <strong>Demo Mode</strong> — No trained model found at <code>saved_models/</code>.
  Showing simulated predictions. Train the model using <code>plant_health_model.py</code> and place the
  <code>.keras</code> file in <code>saved_models/</code> to enable real inference.
</div>
""", unsafe_allow_html=True)


# ── Upload & Predict ────────────────────────────────────────
st.markdown('<div class="section-title">Upload Leaf Image</div>', unsafe_allow_html=True)

upload_col, result_col = st.columns([1, 1], gap="large")

with upload_col:
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded leaf image", use_column_width=True)

        run_btn = st.button("🔍 Analyze Plant Health", use_container_width=True)
    else:
        st.markdown("""
<div style='text-align:center; padding:60px 20px; color:#546e8a; font-size:13px;
            background:#0d1520; border:2px dashed #1e3050; border-radius:10px;'>
  🌿<br><br>
  Supports JPG, PNG, WebP<br>
  Optimal: clear, well-lit leaf photo
</div>
""", unsafe_allow_html=True)
        run_btn = False

with result_col:
    if uploaded and run_btn:
        with st.spinner("Analyzing leaf..."):
            progress = st.progress(0)
            for i in range(0, 101, 10):
                time.sleep(0.04)
                progress.progress(i)

            if model_loaded:
                predictions = real_predict(model, image, top_k=top_k)
            else:
                predictions = demo_predict(image, plant_filter, top_k=top_k)

            report = build_report(predictions)
            progress.empty()

        # ── Status Banner ──
        status_class = "report-healthy" if report["is_healthy"] else "report-diseased"
        tag_class    = "tag-healthy"    if report["is_healthy"] else "tag-diseased"
        status_icon  = "✅"             if report["is_healthy"] else "❌"

        st.markdown(f"""
<div class="report-card {status_class}">
  <div class="report-title">{report['plant']}</div>
  <span class="{tag_class}">{status_icon} {report['status']}</span>
  <div style='margin-top:16px'>
    <div class="info-row">
      <span class="info-label">Disease</span>
      <span class="info-value">{report['disease']}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Confidence</span>
      <span class="info-value">{report['confidence']*100:.1f}%</span>
    </div>
    <div class="info-row">
      <span class="info-label">Severity</span>
      <span class="info-value {report['severity_class']}">{report['severity']}</span>
    </div>
  </div>
  <div class="rec-box">
    💊 <strong>Recommendation:</strong><br>{report['recommendation']}
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Confidence Bar for primary ──
        conf_pct = int(report["confidence"] * 100)
        bar_color = "green" if report["is_healthy"] else "red"
        st.markdown(f"""
<div style='margin-bottom:16px'>
  <div style='display:flex; justify-content:space-between; font-size:11px; color:#546e8a; margin-bottom:4px'>
    <span>Model Confidence</span><span style='color:#e8f4ff'>{conf_pct}%</span>
  </div>
  <div class="conf-bar-wrap">
    <div class="conf-bar-fill-{bar_color}" style='width:{conf_pct}%'></div>
  </div>
</div>
""", unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
<div style='text-align:center; padding:80px 20px; color:#546e8a; font-size:13px;'>
  Results will appear here after analysis
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TOP PREDICTIONS
# ─────────────────────────────────────────────
if uploaded and run_btn and 'predictions' in dir():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top Predictions</div>', unsafe_allow_html=True)

    colors = ["#00e676", "#2979ff", "#ffea00", "#ff9100", "#ff1744",
              "#d500f9", "#00bcd4", "#8bc34a", "#ff5722", "#607d8b"]

    for pred in predictions:
        pct   = pred["confidence"] * 100
        cls   = pred["class"]
        parts = cls.split("___")
        label = f"{parts[0].replace('_',' ')} — {parts[1].replace('_',' ')}" if len(parts) > 1 else cls
        color = colors[pred["rank"] - 1] if pred["rank"] <= len(colors) else "#546e8a"

        st.markdown(f"""
<div class="pred-item">
  <div class="pred-header">
    <span class="pred-name">#{pred['rank']} &nbsp;{label}</span>
    <span class="pred-pct" style='color:{color}'>{pct:.2f}%</span>
  </div>
  <div class="conf-bar-wrap">
    <div style='width:{pct:.1f}%; background:{color}; height:100%; border-radius:4px'></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ATTENTION HEATMAP (Simulated when no model)
# ─────────────────────────────────────────────
if uploaded and run_btn and show_gradcam:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Attention Heatmap (Grad-CAM)</div>', unsafe_allow_html=True)

    try:
        if model_loaded:
            import tensorflow as tf
            from tensorflow import keras

            # Real Grad-CAM
            @st.cache_data
            def compute_gradcam(_model, img_bytes):
                import tensorflow as tf
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(IMG_SIZE)
                arr = np.expand_dims(np.array(img, dtype=np.float32), 0)

                # Find last conv layer
                last_conv = None
                for layer in reversed(_model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv = layer.name
                        break
                if not last_conv:
                    return None

                grad_model = tf.keras.Model(
                    inputs=_model.inputs,
                    outputs=[_model.get_layer(last_conv).output, _model.output]
                )
                with tf.GradientTape() as tape:
                    conv_out, preds = grad_model(arr)
                    idx  = tf.argmax(preds[0])
                    loss = preds[:, idx]

                grads   = tape.gradient(loss, conv_out)
                weights = tf.reduce_mean(grads, axis=(1, 2))
                cam     = tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)
                cam     = tf.maximum(cam, 0).numpy()[0]
                cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                return cam

            img_bytes = uploaded.getvalue()
            heatmap   = compute_gradcam(model, img_bytes)

            if heatmap is not None:
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm

                orig = np.array(image.convert("RGB").resize(IMG_SIZE))
                h_resized = np.array(Image.fromarray(
                    (heatmap * 255).astype(np.uint8)
                ).resize(IMG_SIZE))

                cmap    = cm.get_cmap("jet")
                h_color = (cmap(h_resized / 255.0)[:, :, :3] * 255).astype(np.uint8)
                overlay = (orig * 0.55 + h_color * 0.45).astype(np.uint8)

                hc1, hc2, hc3 = st.columns(3)
                hc1.image(orig,     caption="Original",      use_column_width=True)
                hc2.image(h_color,  caption="Heatmap",       use_column_width=True)
                hc3.image(overlay,  caption="Grad-CAM Overlay", use_column_width=True)
            else:
                st.info("Grad-CAM unavailable for this model architecture.")

        else:
            # Demo heatmap — synthetic gradient visualization
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            orig = np.array(image.convert("RGB").resize(IMG_SIZE)).astype(np.float32)
            H, W = IMG_SIZE

            # Generate synthetic attention map (blob near center-right — typical disease spot)
            y, x  = np.mgrid[0:H, 0:W]
            cy, cx = H * 0.4 + random.randint(-30, 30), W * 0.55 + random.randint(-30, 30)
            sigma  = random.randint(40, 80)
            hmap   = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sigma**2))
            hmap  += 0.15 * np.random.rand(H, W)
            hmap   = np.clip(hmap, 0, 1)

            cmap    = cm.get_cmap("jet")
            h_color = (cmap(hmap)[:, :, :3] * 255).astype(np.uint8)
            overlay = (orig * 0.55 + h_color * 0.45).clip(0, 255).astype(np.uint8)

            hc1, hc2, hc3 = st.columns(3)
            hc1.image(orig.astype(np.uint8),  caption="Original",          use_column_width=True)
            hc2.image(h_color,                 caption="Attention Heatmap", use_column_width=True)
            hc3.image(overlay,                 caption="Overlay (Demo)",    use_column_width=True)

            st.markdown("""
<div style='font-size:11px; color:#546e8a; margin-top:8px'>
  ⚡ Demo heatmap — load a trained model for real Grad-CAM visualizations.
</div>
""", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Heatmap generation failed: {e}")


# ─────────────────────────────────────────────
# BATCH ANALYSIS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div class="section-title">Batch Analysis</div>', unsafe_allow_html=True)

batch_files = st.file_uploader(
    "Upload multiple leaf images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if batch_files:
    if st.button("🌿 Analyze All", use_container_width=False):
        st.markdown(f"**Analyzing {len(batch_files)} images...**")
        cols = st.columns(min(len(batch_files), 4))
        batch_bar = st.progress(0)

        results = []
        for i, f in enumerate(batch_files):
            img = Image.open(f)
            if model_loaded:
                preds = real_predict(model, img, top_k=3)
            else:
                preds = demo_predict(img, "auto", top_k=3)
            r = build_report(preds)
            results.append((f.name, img, r))
            batch_bar.progress(int((i + 1) / len(batch_files) * 100))

        batch_bar.empty()

        cols_per_row = 4
        for row_start in range(0, len(results), cols_per_row):
            row = results[row_start:row_start + cols_per_row]
            cols = st.columns(len(row))
            for col, (fname, img, r) in zip(cols, row):
                with col:
                    col.image(img, use_column_width=True)
                    status_color = "#00e676" if r["is_healthy"] else "#ff1744"
                    col.markdown(f"""
<div style='text-align:center; margin-top:6px'>
  <div style='font-size:11px; color:#546e8a; margin-bottom:2px'>{fname[:20]}</div>
  <div style='font-size:12px; color:{status_color}; font-weight:700'>{r['status']}</div>
  <div style='font-size:11px; color:#c8d8f0'>{r['plant']}</div>
  <div style='font-size:10px; color:#546e8a'>{r['confidence']*100:.1f}% conf.</div>
</div>
""", unsafe_allow_html=True)

        # Summary table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Summary**")
        healthy_count  = sum(1 for _, _, r in results if r["is_healthy"])
        diseased_count = len(results) - healthy_count

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total Analyzed", len(results))
        sc2.metric("Healthy 🌿", healthy_count)
        sc3.metric("Diseased ❌", diseased_count)


# ─────────────────────────────────────────────
# DISEASE ENCYCLOPEDIA
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("📖 Disease Encyclopedia — All 38 Classes"):
    plants = sorted(set(c.split("___")[0] for c in CLASSES))
    plant_tabs = st.tabs(plants)
    for tab, plant in zip(plant_tabs, plants):
        with tab:
            plant_classes = [c for c in CLASSES if c.startswith(plant)]
            for cls in plant_classes:
                disease = cls.split("___")[1].replace("_", " ")
                is_h    = "healthy" in cls.lower()
                dot     = "🟢" if is_h else "🔴"
                st.markdown(f"{dot} **{disease}**")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-top:60px; padding-top:20px;
            border-top:1px solid #1e3050; font-size:10px; color:#1e3050; letter-spacing:2px'>
  PLANT HEALTH CHECKING SYSTEM &nbsp;·&nbsp; CNN + TENSORFLOW &nbsp;·&nbsp; 38 CLASSES
</div>
""", unsafe_allow_html=True)
