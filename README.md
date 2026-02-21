# 👣 FootScan AI — Streamlit App

AI-Enhanced Foot Analysis for Dynamic Size Mapping.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For SAM support (optional but recommended):
```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 2. Run the app

```bash
streamlit run app.py
```

## How it works

1. Upload a photo or use your webcam — foot placed on A4 paper, shot from above.
2. **A4 Detection** — classical edge-based (or SAM if model loaded) finds the sheet.
3. **Pixel→mm calibration** — A4 long side = 297 mm gives the scale factor.
4. **Foot Segmentation** — binary threshold + morphological ops isolates the foot.
5. **Toe Detection** — scanline continuity from bottom of A4 finds the toe tip.
6. **Size Lookup** — foot length (cm) mapped to EU / UK / US / India chart.

## Using the SAM model

1. In the sidebar toggle **"Use SAM model"**.
2. Enter the path to `sam_vit_b_01ec64.pth`.
3. Click **Load SAM** — the model will be cached for the session.

If SAM fails, the app automatically falls back to classical detection.

## Photo Tips

- Place A4 flat on the floor (good lighting).
- Stand with heel touching the short edge of the paper.
- Shoot **straight down** (top-down view).
- Ensure all four corners of the A4 are in frame.
