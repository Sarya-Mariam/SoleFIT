# 👟 AI Foot Size Analyser — Setup Guide

## Project structure
```
shoe_size_app/
├── app.py              ← Streamlit UI
├── shoe_pipeline.py    ← SAM + UNet pipeline logic
├── requirements.txt    ← Python dependencies
├── sam_vit_b_01ec64.pth   ← (you download this)
└── unet_foot.pth          ← (your trained model)
```

---

## 1. Get the model files

### SAM checkpoint
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### UNet checkpoint
Download `unet_foot.pth` from your Google Drive and place it in the same folder.

---

## 2. Install dependencies

```bash
pip install streamlit torch torchvision opencv-python-headless numpy pandas Pillow
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Or install everything at once:
```bash
pip install -r requirements.txt
```

---

## 3. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 4. How to use

1. Set model paths in the **sidebar** (defaults: `sam_vit_b_01ec64.pth`, `unet_foot.pth`)
2. Choose **Upload image** or **Use camera** tab
3. For best results:
   - Place A4 paper flat on the floor
   - Stand with heel touching the **bottom edge** of the A4
   - Photograph from **directly above**
   - Ensure entire A4 is visible in frame
4. Results show **Indian / UK / US / EU** sizes instantly

---

## 5. Google Colab (remote) usage

If you want to run from Colab with your Google Drive models:

```python
# In Colab
from google.colab import drive
drive.mount('/content/drive')

# Copy models locally
import shutil
shutil.copy('/content/drive/MyDrive/YOUR_PATH/unet_foot.pth', 'unet_foot.pth')
shutil.copy('/content/drive/MyDrive/YOUR_PATH/sam_vit_b_01ec64.pth', 'sam_vit_b_01ec64.pth')

# Install & run
!pip install streamlit -q
!pip install git+https://github.com/facebookresearch/segment-anything.git -q
!streamlit run app.py &
# Then use ngrok or localtunnel for the public URL
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| SAM not detected | App falls back to OpenCV A4 detection automatically |
| UNet not loaded | Check file path in sidebar |
| A4 not detected | Ensure paper is flat, fully in frame, good lighting |
| Foot not detected | Ensure foot is clearly visible, not heavily shadowed |
| Wrong size | Try switching size mode (ceil/nearest/floor) in sidebar |
