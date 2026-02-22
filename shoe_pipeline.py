"""
shoe_pipeline.py
Full pipeline: SAM (A4 detection) + UNet (foot segmentation) -> shoe size
Extracted and cleaned from the original Colab notebook.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 1.  UNet model definition
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.u1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.u2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c1 = DoubleConv(1024, 512)
        self.c2 = DoubleConv(512, 256)
        self.c3 = DoubleConv(256, 128)
        self.c4 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b  = self.bottleneck(self.pool(d4))
        u1 = self.c1(torch.cat([self.u1(b),  d4], dim=1))
        u2 = self.c2(torch.cat([self.u2(u1), d3], dim=1))
        u3 = self.c3(torch.cat([self.u3(u2), d2], dim=1))
        u4 = self.c4(torch.cat([self.u4(u3), d1], dim=1))
        return torch.sigmoid(self.out(u4))


# ─────────────────────────────────────────────
# 2.  Shoe size chart  (Converse-style, ceil mode)
# ─────────────────────────────────────────────
_CHART = [
    (21.0,  2.5,  3.0, 35.0),
    (21.5,  3.0,  3.5, 35.5),
    (22.0,  3.5,  4.0, 36.0),
    (22.5,  4.0,  4.5, 37.0),
    (23.0,  4.5,  5.0, 37.5),
    (23.5,  5.0,  5.5, 38.0),
    (24.0,  5.5,  6.5, 39.0),
    (25.0,  6.0,  7.0, 40.0),
    (25.5,  6.5,  7.5, 40.5),
    (26.0,  7.0,  8.0, 41.0),
    (26.5,  7.5,  8.5, 42.0),
    (27.0,  8.0,  9.0, 42.5),
    (27.5,  8.5,  9.5, 43.0),
    (28.0,  9.0, 10.0, 44.0),
    (28.5,  9.5, 10.5, 44.5),
    (29.0, 10.0, 11.0, 45.0),
    (29.5, 10.5, 11.5, 46.0),
    (30.0, 11.0, 12.0, 46.5),
    (30.5, 11.5, 12.5, 47.0),
    (31.0, 12.0, 13.0, 47.5),
    (31.5, 12.5, 13.5, 48.0),
    (32.0, 13.0, 14.0, 49.0),
]

def shoe_size_from_cm(length_cm, mode="ceil"):
    """
    Returns (uk, us, eu) given foot length in cm.
    mode: 'ceil' = next larger size (recommended), 'nearest', 'floor'
    Also returns Indian size (= UK + 1 approx).
    """
    length_cm = round(float(length_cm), 2)
    if mode == "nearest":
        row = min(_CHART, key=lambda r: abs(length_cm - r[0]))
    elif mode == "ceil":
        row = next((r for r in _CHART if r[0] >= length_cm), _CHART[-1])
    else:
        cands = [r for r in _CHART if r[0] <= length_cm]
        row = cands[-1] if cands else _CHART[0]
    cm, uk, us, eu = row
    indian = uk + 1  # approx
    return {"uk": uk, "us": us, "eu": eu, "indian": indian}


# ─────────────────────────────────────────────
# 3.  A4 geometry helpers
# ─────────────────────────────────────────────
def _order_corners(pts):
    pts = np.array(pts)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _rectangularity(cnt):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    return area / (w * h + 1e-6)

def _solidity(cnt):
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return area / (hull_area + 1e-6)

def _edge_straightness(corners):
    edges = [
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
        np.linalg.norm(corners[0] - corners[3]),
    ]
    return min(edges) / (max(edges) + 1e-6)

def _is_a4_contour(cnt, img_area, tol=0.20):
    area = cv2.contourArea(cnt)
    if area < 0.08 * img_area:
        return False, None
    cnt  = cv2.convexHull(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) != 4:
        return False, None
    corners = _order_corners(approx.reshape(4, 2))
    w = np.linalg.norm(corners[1] - corners[0])
    h = np.linalg.norm(corners[3] - corners[0])
    if w < 40 or h < 40:
        return False, None
    ratio = max(w, h) / (min(w, h) + 1e-6)
    if abs(ratio - np.sqrt(2)) > tol:
        return False, None
    if _rectangularity(cnt) < 0.80 or _rectangularity(cnt) > 0.98:
        return False, None
    if _solidity(cnt) < 0.90:
        return False, None
    if _edge_straightness(corners) < 0.65:
        return False, None
    return True, corners

def compute_mm_per_px(corners):
    """corners = [tl, tr, br, bl]  (A4 = 210 x 297 mm)"""
    top_w    = np.linalg.norm(corners[1] - corners[0])
    bottom_w = np.linalg.norm(corners[2] - corners[3])
    px_w = (top_w + bottom_w) / 2.0
    left_h   = np.linalg.norm(corners[3] - corners[0])
    right_h  = np.linalg.norm(corners[2] - corners[1])
    px_h = (left_h + right_h) / 2.0
    mm_per_px_w = 210.0 / (px_w + 1e-6)
    mm_per_px_h = 297.0 / (px_h + 1e-6)
    return (mm_per_px_w + mm_per_px_h) / 2.0


# ─────────────────────────────────────────────
# 4.  SAM-based A4 detection  (with fallback)
# ─────────────────────────────────────────────
def _resize_for_sam(image, max_side=512):
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    return image, scale

def detect_a4_with_sam(image_bgr, predictor):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_small, scale = _resize_for_sam(image_rgb)
    predictor.set_image(image_small)
    h, w, _ = image_small.shape
    img_area = h * w

    box = np.array([0.05*w, 0.05*h, 0.95*w, 0.95*h])
    points = np.array([
        [0.3*w, 0.3*h], [0.7*w, 0.3*h],
        [0.7*w, 0.7*h], [0.3*w, 0.7*h],
    ])
    labels = np.ones(len(points))

    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels,
        box=box, multimask_output=True
    )

    best_score, best_corners = -1, None
    for mask, sam_score in zip(masks, scores):
        mask = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        ok, corners = _is_a4_contour(cnt, img_area)
        if ok:
            area_ratio = cv2.contourArea(cnt) / img_area
            final_score = (
                0.30 * sam_score +
                0.25 * _rectangularity(cnt) +
                0.15 * _solidity(cnt) +
                0.30 * area_ratio
            )
            if final_score > best_score:
                best_score = final_score
                best_corners = corners

    if best_corners is None:
        return None
    return np.array(best_corners / scale, dtype=np.int32)

def detect_a4_fallback(image_bgr):
    """Classic OpenCV fallback when SAM is not available."""
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        ok, corners = _is_a4_contour(cnt, img_area)
        if ok:
            return np.array(corners, dtype=np.int32)
    return None


# ─────────────────────────────────────────────
# 5.  UNet foot segmentation
# ─────────────────────────────────────────────
def unet_segment_foot(image_bgr, model, device, img_size=256, thresh=0.5):
    model.eval()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = image_rgb.shape[:2]
    img_resized = cv2.resize(image_rgb, (img_size, img_size)) / 255.0
    x = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0, 0].detach().cpu().numpy()
    mask_small = (pred > thresh).astype(np.uint8)
    mask_full  = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((5, 5), np.uint8)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask_full


# ─────────────────────────────────────────────
# 6.  Foot length measurement
# ─────────────────────────────────────────────
def _point_to_line_dist(P, A, B):
    AB = B - A
    AP = P - A
    cross = abs(AB[0] * AP[1] - AB[1] * AP[0])
    return cross / (np.linalg.norm(AB) + 1e-6)

def measure_foot_length_px(foot_mask, a4_corners):
    """
    Heel is anchored to the bottom edge of the A4 paper.
    Returns (length_px, toe_point, heel_point, foot_contour).
    """
    cnts, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, None
    foot_cnt = max(cnts, key=cv2.contourArea)
    pts = foot_cnt.reshape(-1, 2).astype(np.float32)

    # Bottom edge of A4
    bl, br = a4_corners[3].astype(np.float32), a4_corners[2].astype(np.float32)

    # Heel = foot point closest to A4 bottom edge
    dists_to_bottom = np.array([_point_to_line_dist(p, bl, br) for p in pts])
    heel_pt = pts[np.argmin(dists_to_bottom)]

    # Toe = foot point farthest from heel
    dists_to_heel = np.linalg.norm(pts - heel_pt, axis=1)
    toe_pt = pts[np.argmax(dists_to_heel)]

    length_px = np.linalg.norm(toe_pt - heel_pt)
    return length_px, toe_pt.astype(int), heel_pt.astype(int), foot_cnt


# ─────────────────────────────────────────────
# 7.  Visualisation helper
# ─────────────────────────────────────────────
def draw_results(image_bgr, a4_corners, foot_mask, toe_pt, heel_pt, foot_cnt, length_cm):
    vis = image_bgr.copy()
    # A4 outline
    if a4_corners is not None:
        pts = a4_corners.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
    # Foot mask overlay
    overlay = vis.copy()
    overlay[foot_mask == 1] = (0, 120, 255)
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    # Foot contour
    if foot_cnt is not None:
        cv2.drawContours(vis, [foot_cnt], -1, (255, 200, 0), 2)
    # Toe-heel line
    if toe_pt is not None and heel_pt is not None:
        cv2.line(vis, tuple(toe_pt), tuple(heel_pt), (255, 50, 50), 3)
        cv2.circle(vis, tuple(toe_pt),  10, (0, 0, 255), -1)
        cv2.circle(vis, tuple(heel_pt), 10, (255, 0, 0), -1)
        mid = ((toe_pt[0] + heel_pt[0]) // 2, (toe_pt[1] + heel_pt[1]) // 2)
        cv2.putText(vis, f"{length_cm:.1f} cm", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return vis


# ─────────────────────────────────────────────
# 8.  Main pipeline class
# ─────────────────────────────────────────────
class ShoeSizePipeline:
    def __init__(self, sam_ckpt_path=None, unet_ckpt_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_predictor = None
        self.unet_model = None

        # Load SAM
        if sam_ckpt_path:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt_path)
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)
                print("✅ SAM loaded")
            except Exception as e:
                print(f"⚠️  SAM not loaded ({e}). Fallback to OpenCV A4 detection.")

        # Load UNet
        if unet_ckpt_path:
            try:
                self.unet_model = UNet().to(self.device)
                self.unet_model.load_state_dict(
                    torch.load(unet_ckpt_path, map_location=self.device)
                )
                self.unet_model.eval()
                print("✅ UNet loaded")
            except Exception as e:
                print(f"⚠️  UNet not loaded ({e}).")

    def predict(self, image_bgr, size_mode="ceil"):
        result = {"ok": False}

        # ── Step 1: detect A4
        if self.sam_predictor is not None:
            a4_corners = detect_a4_with_sam(image_bgr, self.sam_predictor)
        else:
            a4_corners = detect_a4_fallback(image_bgr)

        if a4_corners is None:
            result["error"] = "❌ Could not detect the A4 paper. Make sure the paper is fully visible and flat."
            return result

        mm_per_px = compute_mm_per_px(a4_corners)

        # ── Step 2: segment foot
        if self.unet_model is not None:
            foot_mask = unet_segment_foot(image_bgr, self.unet_model, self.device)
        else:
            result["error"] = "❌ UNet model not loaded. Please provide a valid unet_foot.pth."
            return result

        # ── Step 3: measure length
        length_px, toe_pt, heel_pt, foot_cnt = measure_foot_length_px(foot_mask, a4_corners)
        if length_px is None:
            result["error"] = "❌ Could not detect foot in the image. Ensure the foot is clearly visible."
            return result

        length_mm = length_px * mm_per_px
        length_cm = length_mm / 10.0

        # ── Step 4: size conversion
        sizes     = shoe_size_from_cm(length_cm, mode=size_mode)
        raw_sizes = shoe_size_from_cm(length_cm, mode="floor")

        # ── Step 5: visualise
        vis = draw_results(image_bgr, a4_corners, foot_mask, toe_pt, heel_pt, foot_cnt, length_cm)

        result.update({
            "ok":         True,
            "length_cm":  round(length_cm, 2),
            "length_mm":  round(length_mm, 1),
            "mm_per_px":  round(mm_per_px, 6),
            "final_size": sizes,
            "raw_size":   raw_sizes,
            "vis":        vis,
            "foot_mask":  foot_mask,
            "a4_corners": a4_corners,
        })
        return result
