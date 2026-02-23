"""
app.py – AI Foot Size Analyser  |  Streamlit Cloud deployment
Models are downloaded automatically from Google Drive via gdown on first run,
then cached in the `models/` folder — no re-download on subsequent runs.

👉 ONLY THING TO EDIT: paste your Google Drive share links below.
"""

import os
import zipfile
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ╔══════════════════════════════════════════════════════════════════╗
# ║  🔧  PASTE YOUR GOOGLE DRIVE SHARE LINKS HERE (only edit this)  ║
# ║                                                                  ║
# ║  How to get a share link:                                        ║
# ║  1. Right-click your file in Google Drive → "Share"             ║
# ║  2. Set to "Anyone with the link" → Copy link                   ║
# ║  3. Paste the full link below (the code extracts the file ID)   ║
# ╚══════════════════════════════════════════════════════════════════╝

SAM_GDRIVE_LINK  = "https://drive.google.com/file/d/178OTtYtmm9F9_EorKh3xNHWmPZB_VewS/view?usp=sharing"
UNET_GDRIVE_LINK = "https://drive.google.com/file/d/1LsQwBTEC5G-UlK-hlR_CQlOHM-ykIdUY/view?usp=sharing"

# File names to save as locally (change if your file has a different name)
SAM_FILENAME  = "sam_vit_b_01ec64.pth"   # always a .pth
UNET_FILENAME = "unet_foot.pth"           # direct .pth — no zip needed

# ─────────────────────────────────────────────────────────────────
#  Auto-download + cache logic  (runs once, skips if already cached)
# ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def gdrive_id_from_link(link):
    """Extract file ID from any Google Drive share URL."""
    import re
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
    return m.group(1) if m else None

def download_from_gdrive(link, dest_path, label):
    """Download a file from a public Google Drive link using gdown."""
    import gdown
    file_id = gdrive_id_from_link(link)
    if not file_id:
        st.error(f"❌ Invalid Google Drive link for {label}. Check SAM_GDRIVE_LINK / UNET_GDRIVE_LINK.")
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner(f"⬇️ Downloading {label} from Google Drive (first run only)…"):
        gdown.download(url, dest_path, quiet=False)
    return os.path.exists(dest_path)

def resolve_model(link, filename, label):
    """
    Returns the final .pth path for a model, downloading if needed.
    Handles both direct .pth files and .zip archives.
    Prints debug info to Streamlit logs so errors are always visible.
    """
    try:
        dest     = os.path.join(MODELS_DIR, filename)
        pth_dest = dest.replace(".zip", ".pth")

        # ── Already cached — skip download entirely
        if os.path.exists(pth_dest) and pth_dest.endswith(".pth"):
            print(f"[{label}] Using cached: {pth_dest}")
            return pth_dest
        if os.path.exists(dest) and dest.endswith(".pth"):
            print(f"[{label}] Using cached: {dest}")
            return dest

        # ── Placeholder link — user hasn't filled it in yet
        if "YOUR_" in link:
            print(f"[{label}] Placeholder link — skipping download.")
            return None

        # ── Download
        print(f"[{label}] Downloading from Google Drive…")
        ok = download_from_gdrive(link, dest, label)
        if not ok:
            print(f"[{label}] ❌ Download failed or file missing after download.")
            return None
        print(f"[{label}] Downloaded to: {dest}  ({os.path.getsize(dest)//1024//1024} MB)")

        # ── If it's a zip, extract and find the model file inside
        if filename.endswith(".zip"):
            extract_dir = os.path.join(MODELS_DIR, filename.replace(".zip", "_extracted"))
            os.makedirs(extract_dir, exist_ok=True)

            # First, list ALL files inside the zip before extracting
            with zipfile.ZipFile(dest, "r") as zf:
                zip_contents = zf.namelist()
                print(f"[{label}] ZIP CONTENTS ({len(zip_contents)} files):")
                for name in zip_contents:
                    print(f"[{label}]   {name}")
                print(f"[{label}] Extracting to: {extract_dir}")
                zf.extractall(extract_dir)
            os.remove(dest)  # remove zip to save space

            # Walk extracted dir — accept .pth OR any file matching model name patterns
            MODEL_EXTS = (".pth", ".pt", ".bin", ".ckpt", ".safetensors")
            for root, _, files in os.walk(extract_dir):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    print(f"[{label}] Extracted: {full}")
                    if f.endswith(MODEL_EXTS):
                        print(f"[{label}] ✅ Using model file: {full}")
                        return full

            print(f"[{label}] ❌ No model file found in zip. Contents were: {zip_contents}")
            st.error(f"❌ {label}: No model file found inside zip. See logs for zip contents.")
            return None

        return dest

    except Exception as e:
        import traceback
        print(f"[{label}] ❌ Exception in resolve_model: {e}")
        print(traceback.format_exc())
        st.error(f"❌ {label} model error: {e}")
        return None

# Resolve both models at startup (downloads on first run, cached after)
print("=== Model resolution starting ===")
SAM_PATH  = resolve_model(SAM_GDRIVE_LINK,  SAM_FILENAME,  "SAM")
UNET_PATH = resolve_model(UNET_GDRIVE_LINK, UNET_FILENAME, "UNet")
print(f"SAM_PATH  = {SAM_PATH}")
print(f"UNET_PATH = {UNET_PATH}")
print("=== Model resolution done ===")

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Foot Size Analyser",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── global */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; }

    /* ── header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .header-banner h1 {
        color: #e0e0e0;
        font-size: 2rem;
        margin: 0;
    }
    .header-banner p {
        color: #9ba8c0;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .badge {
        background: #0f3460;
        color: #4fc3f7;
        border: 1px solid #4fc3f7;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
        display: inline-block;
        margin-top: 0.4rem;
    }

    /* ── size cards */
    .size-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .size-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border: 1px solid #2a5298;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .size-card .label {
        color: #7eb3e8;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }
    .size-card .value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .size-card .sub {
        color: #9ba8c0;
        font-size: 0.65rem;
        margin-top: 2px;
    }

    /* ── metric row */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-top: 0.8rem;
    }
    .metric-box {
        flex: 1;
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-box .m-label { color: #7eb3e8; font-size: 0.7rem; letter-spacing: 1px; }
    .metric-box .m-value { color: #e0e0e0; font-size: 1.4rem; font-weight: 700; }

    /* ── info / warning boxes */
    .info-box {
        background: #0d2137;
        border-left: 4px solid #4fc3f7;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #b0c4de;
        margin-top: 0.5rem;
    }
    .warn-box {
        background: #1a1200;
        border-left: 4px solid #f0ad00;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #f0c060;
        margin-top: 0.5rem;
    }

    /* ── step indicator */
    .step-row { display: flex; gap: 0.5rem; margin: 1rem 0; }
    .step { flex: 1; text-align: center; padding: 0.5rem 0.3rem;
            border-radius: 8px; font-size: 0.75rem; font-weight: 600; }
    .step.done  { background: #0a3d0a; color: #4caf50; border: 1px solid #4caf50; }
    .step.active{ background: #0f3460; color: #4fc3f7; border: 1px solid #4fc3f7; }
    .step.wait  { background: #111; color: #555; border: 1px solid #333; }

    /* sidebar */
    section[data-testid="stSidebar"] { background: #0d1b2a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div style="font-size:3rem;">👟</div>
  <div>
    <h1>AI Foot Size Analyser</h1>
    <p>SAM-powered A4 calibration &nbsp;·&nbsp; U-Net foot segmentation &nbsp;·&nbsp; Multi-chart sizing</p>
    <span class="badge">AI ENHANCED</span>
    <span class="badge" style="margin-left:6px;">DYNAMIC SIZE MAPPING</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar – model paths & settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ── MODEL STATUS ──────────────────────────────────────────────
    st.markdown("### 🗂️ Model status")
    st.caption("Models are auto-downloaded from Google Drive on first run and cached.")

    # SAM
    if SAM_PATH and os.path.exists(SAM_PATH):
        st.success(f"✅ SAM — cached ✓")
        st.caption(f"`{os.path.basename(SAM_PATH)}`")
    elif "YOUR_" in SAM_GDRIVE_LINK:
        st.warning("⚙️ SAM — paste your Drive link in `app.py` → `SAM_GDRIVE_LINK`")
    else:
        st.error("❌ SAM — download failed. Check the Drive link is public.")

    # UNet
    if UNET_PATH and os.path.exists(UNET_PATH):
        st.success(f"✅ UNet — cached ✓")
        st.caption(f"`{os.path.basename(UNET_PATH)}`")
    elif "YOUR_" in UNET_GDRIVE_LINK:
        st.warning("⚙️ UNet — paste your Drive link in `app.py` → `UNET_GDRIVE_LINK`")
    else:
        st.error("❌ UNet — download failed. Check the Drive link is public.")

    st.divider()
    st.markdown("### 🔗 Setup (one-time)")
    st.markdown("""
1. Open your file in **Google Drive**
2. Click **Share** → set to **Anyone with the link**
3. Copy the link and paste into `app.py`:
   - `SAM_GDRIVE_LINK`
   - `UNET_GDRIVE_LINK`
4. Push to GitHub → Streamlit Cloud redeploys automatically
    """)

    # ── INFERENCE SETTINGS ────────────────────────────────────────
    st.divider()
    st.markdown("### 🛠️ Inference settings")
    size_mode = st.radio(
        "Size recommendation mode",
        options=["ceil", "nearest", "floor"],
        index=0,
        help="ceil = next larger size (recommended to avoid short fit)"
    )
    jack_purcell = st.checkbox(
        "Jack Purcell / Chuck 70 mode (runs large — half size down)",
        value=False,
        help="Enable if using Converse Jack Purcell or Chuck 70 which run large"
    )
    unet_thresh = st.slider(
        "UNet mask threshold", 0.1, 0.9, 0.5, 0.05,
        help="Lower = include more foot area; raise if background leaks into mask"
    )

    st.divider()
    st.markdown("### 📸 How to capture")
    st.markdown("""
1. Place **A4 paper flat** on the floor
2. Stand with **heel touching** the paper's bottom edge
3. Capture from **directly above** (parallel to floor)
4. Ensure **good lighting** — avoid shadows over foot
5. Keep the **entire A4 visible** in frame
    """)


# ─────────────────────────────────────────────
#  Load pipeline (cached)
# ─────────────────────────────────────────────
# Pipeline only stores checkpoint paths — models are loaded/unloaded per-request
# This keeps RAM under Streamlit Cloud's 1GB limit (SAM alone = 357MB)
@st.cache_resource(show_spinner=False)
def load_pipeline(sam_ckpt, unet_ckpt):
    from shoe_pipeline import ShoeSizePipeline
    # No models loaded into RAM here — just paths stored
    return ShoeSizePipeline(
        sam_ckpt_path  = sam_ckpt  if sam_ckpt  and os.path.exists(sam_ckpt)  else None,
        unet_ckpt_path = unet_ckpt if unet_ckpt and os.path.exists(unet_ckpt) else None,
    )

if SAM_PATH or UNET_PATH:
    pipeline = load_pipeline(SAM_PATH, UNET_PATH)
    sam_ok  = bool(pipeline.sam_predictor)
    unet_ok = bool(pipeline.unet_model)
else:
    pipeline = None
    sam_ok   = False
    unet_ok  = False

# Model status banner
col_s1, col_s2 = st.columns(2)
with col_s1:
    if sam_ok:
        st.success("✅ SAM model loaded (A4 detection)")
    else:
        st.warning("⚠️ SAM not found — using OpenCV fallback for A4 detection")
with col_s2:
    if unet_ok:
        st.success("✅ UNet model loaded (foot segmentation)")
    else:
        st.error("❌ UNet model not loaded — predictions unavailable")


# ─────────────────────────────────────────────
#  Input section – upload OR camera
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📷 Provide foot image")

tab_upload, tab_camera = st.tabs(["📂 Upload image", "📷 Use camera"])

image_bgr = None

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a foot-on-A4 photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

with tab_camera:
    cam_img = st.camera_input("Take a photo of your foot on the A4 paper")
    if cam_img:
        arr = np.frombuffer(cam_img.read(), np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─────────────────────────────────────────────
#  Run pipeline
# ─────────────────────────────────────────────
if image_bgr is not None:
    st.markdown("---")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Original image")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), width='stretch')

    # ── pipeline steps indicator
    st.markdown("""
    <div class="step-row">
      <div class="step done">✔ Image loaded</div>
      <div class="step active">⟳ A4 detection</div>
      <div class="step wait">○ Foot segment</div>
      <div class="step wait">○ Size mapping</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running SAM + UNet pipeline…"):
        if pipeline is None:
            st.error("⚠️ Please upload both model files in the sidebar before running.")
            st.stop()
        result = pipeline.predict(image_bgr, mode=size_mode, jack_purcell=jack_purcell, unet_thresh=unet_thresh)

    if not result["ok"]:
        st.error(result.get("error", "Unknown error"))
    else:
        # Update steps
        st.markdown("""
        <div class="step-row">
          <div class="step done">✔ Image loaded</div>
          <div class="step done">✔ A4 detected</div>
          <div class="step done">✔ Foot segmented</div>
          <div class="step done">✔ Size mapped</div>
        </div>
        """, unsafe_allow_html=True)

        with col_right:
            st.markdown("#### Analysis output")
            st.image(cv2.cvtColor(result["vis"], cv2.COLOR_BGR2RGB), width='stretch')

        # ── metrics row
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box">
            <div class="m-label">FOOT LENGTH</div>
            <div class="m-value">{result['length_cm']:.1f} cm</div>
          </div>
          <div class="metric-box">
            <div class="m-label">FOOT LENGTH</div>
            <div class="m-value">{result['length_mm']:.0f} mm</div>
          </div>
          <div class="metric-box">
            <div class="m-label">MM / PIXEL</div>
            <div class="m-value">{result['mm_per_px']:.4f}</div>
          </div>
          <div class="metric-box">
            <div class="m-label">SIZE MODE</div>
            <div class="m-value" style="font-size:1.1rem;">{size_mode.upper()}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── size cards  (Converse chart: uk, us_men, us_women, eu, indian)
        st.markdown("#### 🥿 Recommended shoe size")
        fs = result["final_size"]
        rs = result["raw_size"]
        # CSS grid now needs 5 columns
        st.markdown("""
        <style>
        .size-grid { grid-template-columns: repeat(5, 1fr) !important; }
        </style>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="size-grid">
          <div class="size-card">
            <div class="label">🇮🇳 Indian</div>
            <div class="value">{fs['indian']}</div>
            <div class="sub">raw: {rs['indian']}</div>
          </div>
          <div class="size-card">
            <div class="label">🇬🇧 UK</div>
            <div class="value">{fs['uk']}</div>
            <div class="sub">raw: {rs['uk']}</div>
          </div>
          <div class="size-card">
            <div class="label">🇺🇸 US Men</div>
            <div class="value">{fs['us_men']}</div>
            <div class="sub">raw: {rs['us_men']}</div>
          </div>
          <div class="size-card">
            <div class="label">🇺🇸 US Women</div>
            <div class="value">{fs['us_women']}</div>
            <div class="sub">raw: {rs['us_women']}</div>
          </div>
          <div class="size-card">
            <div class="label">🇪🇺 EU</div>
            <div class="value">{fs['eu']}</div>
            <div class="sub">raw: {rs['eu']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
          💡 <strong>Recommended size</strong> uses the <em>next larger</em> size (ceil mode) from
          the Converse official chart to avoid a short fit. The <em>raw</em> value shows the exact match.
          Enable <strong>Jack Purcell mode</strong> in the sidebar if the shoe model runs large.
        </div>
        """, unsafe_allow_html=True)

        # ── foot mask visualisation expander
        with st.expander("🔬 View segmentation masks"):
            c1, c2 = st.columns(2)
            foot_mask_vis = (result["foot_mask"] * 255).astype(np.uint8)
            foot_mask_rgb = cv2.cvtColor(foot_mask_vis, cv2.COLOR_GRAY2RGB)
            a4_vis = image_bgr.copy()
            if result["a4_corners"] is not None:
                pts = result["a4_corners"].reshape(-1, 1, 2)
                cv2.polylines(a4_vis, [pts], True, (0, 255, 0), 4)
            with c1:
                st.markdown("**A4 detection**")
                st.image(cv2.cvtColor(a4_vis, cv2.COLOR_BGR2RGB), width='stretch')
            with c2:
                st.markdown("**Foot mask (UNet)**")
                st.image(foot_mask_rgb, width='stretch')

        # ── download result image
        _, buf = cv2.imencode(".jpg", result["vis"])
        st.download_button(
            label="⬇️ Download result image",
            data=buf.tobytes(),
            file_name="foot_analysis_result.jpg",
            mime="image/jpeg",
        )

elif not (image_bgr is not None):
    st.markdown("""
    <div class="info-box">
      👆 Upload a photo or use your camera to get started.<br>
      Place your foot on an A4 sheet of paper and photograph from directly above.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Size chart reference (collapsible)
# ─────────────────────────────────────────────
with st.expander("📏 Full size chart reference"):
    import pandas as pd
    chart_data = [
        (21.0, 2.5, 3.0, 35.0, 3.5), (21.5, 3.0, 3.5, 35.5, 4.0),
        (22.0, 3.5, 4.0, 36.0, 4.5), (22.5, 4.0, 4.5, 37.0, 5.0),
        (23.0, 4.5, 5.0, 37.5, 5.5), (23.5, 5.0, 5.5, 38.0, 6.0),
        (24.0, 5.5, 6.5, 39.0, 6.5), (25.0, 6.0, 7.0, 40.0, 7.0),
        (25.5, 6.5, 7.5, 40.5, 7.5), (26.0, 7.0, 8.0, 41.0, 8.0),
        (26.5, 7.5, 8.5, 42.0, 8.5), (27.0, 8.0, 9.0, 42.5, 9.0),
        (27.5, 8.5, 9.5, 43.0, 9.5), (28.0, 9.0, 10.0, 44.0, 10.0),
        (28.5, 9.5, 10.5, 44.5, 10.5), (29.0, 10.0, 11.0, 45.0, 11.0),
        (29.5, 10.5, 11.5, 46.0, 11.5), (30.0, 11.0, 12.0, 46.5, 12.0),
    ]
    df = pd.DataFrame(chart_data, columns=["Foot length (cm)", "UK", "US", "EU", "Indian (approx)"])
    st.dataframe(df, width='stretch', hide_index=True)


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("AI Enhanced Foot Analysis · SAM (Meta) + U-Net · Built with Streamlit")
