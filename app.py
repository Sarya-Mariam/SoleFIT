"""
AI-Enhanced Foot Analysis for Dynamic Size Mapping
Streamlit Web App
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FootScan AI",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — clean dark-clinical aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161a22;
    --border: #252c3a;
    --accent: #00e5a0;
    --accent2: #0099ff;
    --warn: #ff6b35;
    --text: #e2e8f0;
    --muted: #64748b;
    --radius: 12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: clamp(2rem, 5vw, 3.5rem);
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Metric chips */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}
.metric-chip {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
}
.metric-chip .label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-chip .value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1.1;
}
.metric-chip .sub {
    font-size: 0.75rem;
    color: var(--muted);
}

/* Indian size chip */
.metric-chip.india .value { color: #ff9f43; }
.metric-chip.uk    .value { color: var(--accent); }
.metric-chip.eu    .value { color: var(--accent2); }
.metric-chip.us    .value { color: #a29bfe; }
.metric-chip.cm    .value { color: #fd79a8; }

/* Status badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
}
.badge.ok  { background: rgba(0,229,160,.15); color: var(--accent); border: 1px solid var(--accent); }
.badge.err { background: rgba(255,107,53,.15); color: var(--warn);   border: 1px solid var(--warn); }

/* Steps indicator */
.steps {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 1rem;
}
.step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--border);
}
.step-dot.active { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.step-dot.done   { background: var(--accent2); }

/* Upload zone override */
[data-testid="stFileUploader"] > div {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.6rem 2rem !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,229,160,.4);
}

/* Instruction list */
.instruction-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.instruction-list li {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-size: 0.9rem;
}
.instruction-list li:last-child { border-bottom: none; }
.instruction-list .num {
    background: var(--accent);
    color: #000;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    width: 22px; height: 22px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem;
    flex-shrink: 0;
    margin-top: 1px;
}

/* Divider */
.divider { height: 1px; background: var(--border); margin: 1.5rem 0; }

/* Info box */
.info-box {
    background: rgba(0,153,255,.08);
    border-left: 3px solid var(--accent2);
    padding: 0.75rem 1rem;
    border-radius: 0 var(--radius) var(--radius) 0;
    font-size: 0.85rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIZE CHART  (foot-length cm → EU / UK / US / India)
# ─────────────────────────────────────────────────────────────────────────────
SIZE_CHART = [
    {"cm": 21.5, "EU": 34, "UK": 2,   "US": 3,   "IN": 1},
    {"cm": 22.0, "EU": 35, "UK": 2.5, "US": 4,   "IN": 2},
    {"cm": 22.5, "EU": 36, "UK": 3,   "US": 4.5, "IN": 3},
    {"cm": 23.0, "EU": 37, "UK": 4,   "US": 5,   "IN": 4},
    {"cm": 23.5, "EU": 38, "UK": 5,   "US": 6,   "IN": 5},
    {"cm": 24.0, "EU": 39, "UK": 5.5, "US": 6.5, "IN": 6},
    {"cm": 24.5, "EU": 40, "UK": 6,   "US": 7,   "IN": 7},
    {"cm": 25.0, "EU": 41, "UK": 6.5, "US": 7.5, "IN": 8},
    {"cm": 25.5, "EU": 42, "UK": 7,   "US": 8,   "IN": 9},
    {"cm": 26.0, "EU": 43, "UK": 8,   "US": 9,   "IN": 10},
    {"cm": 26.5, "EU": 44, "UK": 8.5, "US": 9.5, "IN": 11},
    {"cm": 27.0, "EU": 45, "UK": 9,   "US": 10,  "IN": 12},
    {"cm": 27.5, "EU": 46, "UK": 10,  "US": 11,  "IN": 13},
    {"cm": 28.0, "EU": 47, "UK": 11,  "US": 12,  "IN": 14},
    {"cm": 28.5, "EU": 48, "UK": 12,  "US": 13,  "IN": 15},
]

def cm_to_sizes(cm: float) -> dict:
    entry = min(SIZE_CHART, key=lambda x: abs(x["cm"] - cm))
    return entry


# ─────────────────────────────────────────────────────────────────────────────
# A4 DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl])


def rectangularity(cnt):
    area = cv2.contourArea(cnt)
    _, _, w, h = cv2.boundingRect(cnt)
    return area / (w * h + 1e-6)


def solidity(cnt):
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return area / (hull_area + 1e-6)


def is_a4_like(cnt, img_area, tol=0.22):
    """Returns (ok, corners) — checks area, aspect-ratio, rectangularity."""
    area = cv2.contourArea(cnt)
    if area < 0.05 * img_area:
        return False, None

    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
    if len(approx) != 4:
        return False, None

    corners = order_corners(approx.reshape(4, 2))
    w = np.linalg.norm(corners[1] - corners[0])
    h = np.linalg.norm(corners[3] - corners[0])
    if w < 30 or h < 30:
        return False, None

    ratio = max(w, h) / (min(w, h) + 1e-6)
    if abs(ratio - np.sqrt(2)) > tol:
        return False, None

    if rectangularity(hull) < 0.75:
        return False, None
    if solidity(hull) < 0.85:
        return False, None

    return True, corners


def detect_a4_classical(image_bgr: np.ndarray):
    """
    Purely classical (no SAM) A4 detection.
    Works for standard white-paper-on-floor setups.
    Returns corners (4×2) or None.
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w

    gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (7, 7), 0)
    edges  = cv2.Canny(blur, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best_score   = -1
    best_corners = None

    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:12]:
        ok, corners = is_a4_like(cnt, img_area)
        if ok:
            score = cv2.contourArea(cnt) / img_area
            if score > best_score:
                best_score   = score
                best_corners = corners

    return best_corners


def detect_a4_sam(image_bgr: np.ndarray, predictor):
    """SAM-based A4 detection (used when SAM model is loaded)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    img_area = h * w

    predictor.set_image(image_rgb)

    box = np.array([[0.03*w, 0.03*h, 0.97*w, 0.97*h]])
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)

    if masks is None or len(masks) == 0:
        return None

    best_corners = None
    best_score   = -1

    for mask, sam_sc in zip(masks, scores):
        m = mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        ok, corners = is_a4_like(cnt, img_area)
        if ok:
            final = 0.4 * sam_sc + 0.6 * (cv2.contourArea(cnt) / img_area)
            if final > best_score:
                best_score   = final
                best_corners = corners

    return best_corners


# ─────────────────────────────────────────────────────────────────────────────
# FOOT MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────
def measure_foot(image_bgr: np.ndarray, a4_corners: np.ndarray):
    """
    Robust foot measurement:
    - Binary threshold → morphological clean
    - Scanline from bottom → first continuous band = toe
    - PCA toe detection as fallback
    Returns (vis_rgb, foot_cm, foot_mask_rgb) or raises RuntimeError.
    """
    a4_height_px = np.linalg.norm(a4_corners[3] - a4_corners[0])
    mm_per_px    = 297.0 / a4_height_px

    a4_top_y = int(np.min(a4_corners[:, 1]))
    a4_bot_y = int(np.max(a4_corners[:, 1]))
    a4_x_min = int(np.min(a4_corners[:, 0]))
    a4_x_max = int(np.max(a4_corners[:, 0]))

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    # Remove area below A4 bottom
    mask[a4_bot_y:, :] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ── Scanline toe detection ──
    min_width = int(0.20 * (a4_x_max - a4_x_min))
    toe_y = None
    toe_x = None
    inside_foot = False

    for y in range(a4_bot_y, a4_top_y, -1):
        row = mask[y, a4_x_min:a4_x_max]
        run = best_run = 0
        xs  = best_xs  = []
        for i, val in enumerate(row):
            if val == 255:
                run += 1
                xs.append(i)
                if run > best_run:
                    best_run = run
                    best_xs  = xs.copy()
            else:
                run = 0
                xs  = []
        if best_run >= min_width:
            inside_foot = True
            toe_y = y
            toe_x = int(a4_x_min + np.mean(best_xs))
        else:
            if inside_foot:
                break

    if toe_y is None:
        raise RuntimeError("Toe not detected — make sure the foot is clearly visible on the A4 paper.")

    dist_px = abs(toe_y - a4_top_y)
    foot_cm = round((dist_px * mm_per_px) / 10.0, 2)

    # ── Visualisation ──
    vis = image_rgb.copy()
    cv2.polylines(vis, [a4_corners.reshape(-1,1,2).astype(int)], True, (0,229,160), 3)
    cv2.line(vis, (toe_x, a4_top_y), (toe_x, toe_y), (255, 80, 80), 3)
    cv2.circle(vis, (toe_x, a4_top_y), 8, (255, 0, 255), -1)
    cv2.circle(vis, (toe_x, toe_y),    8, (0,   0, 255), -1)

    # foot mask overlay (coloured)
    foot_vis = image_rgb.copy()
    overlay  = np.zeros_like(foot_vis)
    overlay[mask == 255] = [0, 229, 160]
    foot_vis = cv2.addWeighted(foot_vis, 0.7, overlay, 0.3, 0)

    return vis, foot_cm, foot_vis, mask


# ─────────────────────────────────────────────────────────────────────────────
# SAM MODEL LOADER  (optional, cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_sam_model(checkpoint_path: str):
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: bytes / PIL → BGR numpy
# ─────────────────────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(image_bgr: np.ndarray, predictor):
    """Returns dict with all results or raises RuntimeError."""

    # 1. A4 detection
    if predictor is not None:
        a4_corners = detect_a4_sam(image_bgr, predictor)
        method = "SAM"
        if a4_corners is None:
            a4_corners = detect_a4_classical(image_bgr)
            method = "Classical (SAM fallback)"
    else:
        a4_corners = detect_a4_classical(image_bgr)
        method = "Classical"

    if a4_corners is None:
        raise RuntimeError(
            "A4 paper not detected. "
            "Make sure the entire sheet is visible, well-lit, and flat."
        )

    # 2. Foot measurement
    vis, foot_cm, foot_vis, foot_mask = measure_foot(image_bgr, a4_corners)

    # 3. Size conversion
    sizes = cm_to_sizes(foot_cm)

    return {
        "a4_corners": a4_corners,
        "vis":        vis,
        "foot_vis":   foot_vis,
        "foot_mask":  foot_mask,
        "foot_cm":    foot_cm,
        "sizes":      sizes,
        "method":     method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI — HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>👣 FootScan AI</h1>
  <p>AI-Enhanced Foot Analysis · Dynamic Size Mapping · SAM + Computer Vision</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Model config
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    use_sam = st.toggle("Use SAM model", value=False,
                        help="Enable if you have the SAM checkpoint downloaded.")
    sam_path = ""
    predictor = None

    if use_sam:
        sam_path = st.text_input(
            "SAM checkpoint path",
            value="sam_vit_b_01ec64.pth",
            help="Absolute or relative path to the SAM ViT-B checkpoint."
        )
        if st.button("Load SAM"):
            with st.spinner("Loading SAM model…"):
                predictor = load_sam_model(sam_path)
            if predictor:
                st.success("SAM loaded ✓")
                st.session_state["predictor"] = predictor
            else:
                st.error("Failed to load SAM. Check path & dependencies.")
        else:
            predictor = st.session_state.get("predictor", None)

    st.markdown("---")
    st.markdown("### 📋 How It Works")
    st.markdown("""
1. **A4 Detection** — SAM or classical edge-based segmentation finds the paper
2. **Pixel → mm calibration** — A4 long side = 297 mm
3. **Foot Segmentation** — Binary threshold + morphological cleaning
4. **Toe Detection** — Scanline continuity from bottom
5. **Size Lookup** — EU / UK / US / India chart
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Instructions
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📌 How to take a good photo", expanded=False):
    st.markdown("""
<ul class="instruction-list">
  <li><span class="num">1</span>Place an A4 paper flat on the floor.</li>
  <li><span class="num">2</span>Stand with your foot on the paper (heel touching the short edge).</li>
  <li><span class="num">3</span>Take a top-down photo — camera directly above.</li>
  <li><span class="num">4</span>Ensure all four edges of the A4 are clearly visible.</li>
  <li><span class="num">5</span>Good lighting, no heavy shadows over the paper.</li>
</ul>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT — Upload or Camera
# ─────────────────────────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Take a Photo"])

image_bgr = None

with tab_upload:
    uploaded = st.file_uploader(
        "Drop your foot image here (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        image_bgr = bytes_to_bgr(uploaded.read())

with tab_camera:
    camera_img = st.camera_input("Take a photo of your foot on A4 paper")
    if camera_img:
        image_bgr = bytes_to_bgr(camera_img.read())


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if image_bgr is not None:

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_orig, col_run = st.columns([2, 1])
    with col_orig:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                 caption="Input image", use_container_width=True)
    with col_run:
        st.markdown("### Ready to analyse")
        st.markdown(
            f'<div class="info-box">Detection engine: '
            f'{"SAM + Classical fallback" if predictor else "Classical (edge-based)"}</div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        run_btn = st.button("🔍 Analyse Foot", use_container_width=True)

    if run_btn:
        with st.spinner("Analysing… detecting A4 → measuring foot…"):
            try:
                result = run_pipeline(image_bgr, predictor)
            except RuntimeError as e:
                st.markdown(
                    f'<div class="card"><span class="badge err">❌ Error</span>'
                    f'<p style="margin-top:.75rem;color:#ff6b35">{e}</p></div>',
                    unsafe_allow_html=True
                )
                st.stop()

        foot_cm = result["foot_cm"]
        sizes   = result["sizes"]

        # ── Size Results ────────────────────────────────────────────────────
        st.markdown("""
<div class="card">
  <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;">
    <span style="font-size:1.5rem">📏</span>
    <h3 style="margin:0">Measurement Results</h3>
    <span class="badge ok">✓ Success</span>
  </div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="metrics-row">
  <div class="metric-chip cm">
    <div class="label">Foot Length</div>
    <div class="value">{foot_cm}</div>
    <div class="sub">centimetres</div>
  </div>
  <div class="metric-chip eu">
    <div class="label">EU Size</div>
    <div class="value">{sizes["EU"]}</div>
    <div class="sub">European</div>
  </div>
  <div class="metric-chip uk">
    <div class="label">UK Size</div>
    <div class="value">{sizes["UK"]}</div>
    <div class="sub">British</div>
  </div>
  <div class="metric-chip us">
    <div class="label">US Size</div>
    <div class="value">{sizes["US"]}</div>
    <div class="sub">American</div>
  </div>
  <div class="metric-chip india">
    <div class="label">India Size</div>
    <div class="value">{sizes["IN"]}</div>
    <div class="sub">Indian</div>
  </div>
</div>
<div class="info-box" style="margin-top:1rem">
  Detection method: {result["method"]} &nbsp;|&nbsp;
  A4 long-side pixel height used for calibration (297 mm reference)
</div>
</div>
""", unsafe_allow_html=True)

        # ── Visual Results ───────────────────────────────────────────────────
        st.markdown("### 🖼️ Visual Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.image(result["vis"],
                     caption="A4 detected (green) + measurement line (red)",
                     use_container_width=True)
        with c2:
            st.image(result["foot_vis"],
                     caption="Foot segmentation overlay",
                     use_container_width=True)

        # ── Full Size Chart ──────────────────────────────────────────────────
        with st.expander("📊 Full Shoe Size Reference Chart"):
            import pandas as pd
            df = pd.DataFrame(SIZE_CHART)
            df.columns = ["Foot Length (cm)", "EU", "UK", "US", "India"]

            def highlight_row(row):
                if abs(row["Foot Length (cm)"] - foot_cm) < 0.3:
                    return ["background-color: rgba(0,229,160,.15); color: #00e5a0"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df.style.apply(highlight_row, axis=1),
                use_container_width=True,
                hide_index=True,
            )

        # ── Download measurement image ───────────────────────────────────────
        vis_pil  = Image.fromarray(result["vis"])
        buf = io.BytesIO()
        vis_pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Measurement Image",
            data=buf.getvalue(),
            file_name="footscan_result.png",
            mime="image/png",
        )

else:
    # Empty state
    st.markdown("""
<div class="card" style="text-align:center;padding:3rem;">
  <div style="font-size:4rem;margin-bottom:1rem">👣</div>
  <h3 style="color:var(--muted)">Upload or capture a foot image to begin</h3>
  <p style="color:var(--muted);font-family:'DM Mono',monospace;font-size:.8rem">
    Place your foot on an A4 sheet and take a clear top-down photo
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1rem;
            color:var(--muted);font-family:'DM Mono',monospace;font-size:.75rem">
  FootScan AI — SAM · Computer Vision · Dynamic Size Mapping
</div>
""", unsafe_allow_html=True)
