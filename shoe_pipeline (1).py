"""
shoe_pipeline.py
Exact notebook methodology + memory-optimised for Streamlit Cloud (1GB RAM limit).
Clears GPU/CPU tensors and runs gc.collect() after each heavy step.
"""

import gc
import os
import cv2
import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════
#  Memory helper
# ══════════════════════════════════════════════════════════════

def free_memory():
    """Force garbage collection and clear torch cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
#  1.  UNet architecture  (exact notebook)
# ══════════════════════════════════════════════════════════════

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
        self.u2 = nn.ConvTranspose2d(512,  256, 2, stride=2)
        self.u3 = nn.ConvTranspose2d(256,  128, 2, stride=2)
        self.u4 = nn.ConvTranspose2d(128,   64, 2, stride=2)
        self.c1 = DoubleConv(1024, 512)
        self.c2 = DoubleConv(512,  256)
        self.c3 = DoubleConv(256,  128)
        self.c4 = DoubleConv(128,   64)
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


# ══════════════════════════════════════════════════════════════
#  2.  Shoe size chart  (Converse official — notebook cell 24)
#      row: (foot_len_cm, us_men, us_women, uk, eu)
# ══════════════════════════════════════════════════════════════

_CONVERSE_CHART = [
    (21.0,  3.0,  4.5,  2.5, 35.0),
    (21.5,  3.5,  5.0,  3.0, 35.5),
    (22.0,  4.0,  5.5,  3.5, 36.0),
    (22.5,  4.5,  6.0,  4.0, 37.0),
    (23.0,  5.0,  6.5,  4.5, 37.5),
    (23.5,  5.5,  7.0,  5.0, 38.0),
    (24.0,  6.5,  8.0,  5.5, 39.0),
    (25.0,  7.0,  8.5,  6.0, 40.0),
    (25.5,  7.5,  9.0,  6.5, 40.5),
    (26.0,  8.0,  9.5,  7.0, 41.0),
    (26.5,  8.5, 10.0,  7.5, 42.0),
    (27.0,  9.0, 10.5,  8.0, 42.5),
    (27.5,  9.5, 11.0,  8.5, 43.0),
    (28.0, 10.0, 11.5,  9.0, 44.0),
    (28.5, 10.5, 12.0,  9.5, 44.5),
    (29.0, 11.0, 12.5, 10.0, 45.0),
    (29.5, 11.5, 13.0, 10.5, 46.0),
    (30.0, 12.0, 13.5, 11.0, 46.5),
    (30.5, 12.5, 14.0, 11.5, 47.0),
    (31.0, 13.0, 14.5, 12.0, 47.5),
    (31.5, 13.5, 15.0, 12.5, 48.0),
    (32.0, 14.0, 15.5, 13.0, 49.0),
]


def shoe_size_from_foot_length_cm(length_cm, mode="ceil", jack_purcell=False):
    length_cm = round(float(length_cm), 2)
    if mode == "nearest":
        row = min(_CONVERSE_CHART, key=lambda r: abs(length_cm - r[0]))
    elif mode == "ceil":
        row = next((r for r in _CONVERSE_CHART if r[0] >= length_cm), _CONVERSE_CHART[-1])
    else:
        cands = [r for r in _CONVERSE_CHART if r[0] <= length_cm]
        row = cands[-1] if cands else _CONVERSE_CHART[0]
    _, us_men, us_women, uk, eu = row
    if jack_purcell:
        us_men -= 0.5; us_women -= 0.5; uk -= 0.5
    return {"uk": uk, "us_men": us_men, "us_women": us_women, "eu": eu, "indian": uk + 1}


# ══════════════════════════════════════════════════════════════
#  3.  A4 detection helpers  (exact notebook)
# ══════════════════════════════════════════════════════════════

def resize_for_sam(image, max_side=512):
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    return image, scale


def order_corners(pts):
    pts  = np.array(pts)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]])


def rectangularity(cnt):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    return area / (w * h + 1e-6)


def solidity(cnt):
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return area / (hull_area + 1e-6)


def edge_straightness(corners):
    edges = [np.linalg.norm(corners[(i+1)%4] - corners[i]) for i in range(4)]
    return min(edges) / (max(edges) + 1e-6)


def is_a4_contour(cnt, img_area, tol=0.20):
    if cv2.contourArea(cnt) < 0.08 * img_area:
        return False, None
    cnt  = cv2.convexHull(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) != 4:
        return False, None
    corners = order_corners(approx.reshape(4, 2))
    w = np.linalg.norm(corners[1] - corners[0])
    h = np.linalg.norm(corners[3] - corners[0])
    if w < 40 or h < 40:
        return False, None
    if abs(max(w,h) / (min(w,h) + 1e-6) - np.sqrt(2)) > tol:
        return False, None
    rect = rectangularity(cnt)
    if rect < 0.80 or rect > 0.98:
        return False, None
    if solidity(cnt) < 0.90:
        return False, None
    if edge_straightness(corners) < 0.65:
        return False, None
    return True, corners


def detect_a4_with_sam(image_bgr, predictor):
    """
    SAM A4 detection — exact notebook logic.
    Resizes image to 512px before passing to SAM to save memory.
    """
    # Downscale input image to save memory before SAM processing
    h0, w0 = image_bgr.shape[:2]
    max_input = 1024  # cap input resolution
    scale_input = min(1.0, max_input / max(h0, w0))
    if scale_input < 1.0:
        image_bgr = cv2.resize(image_bgr,
                               (int(w0 * scale_input), int(h0 * scale_input)),
                               interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_small, scale = resize_for_sam(image_rgb)
    del image_rgb
    free_memory()

    predictor.set_image(image_small)
    h, w, _ = image_small.shape
    img_area = h * w
    del image_small
    free_memory()

    box    = np.array([0.05*w, 0.05*h, 0.95*w, 0.95*h])
    points = np.array([[0.3*w, 0.3*h], [0.7*w, 0.3*h],
                       [0.7*w, 0.7*h], [0.3*w, 0.7*h]])
    labels = np.ones(len(points))

    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels,
        box=box, multimask_output=True
    )
    free_memory()

    best_score, best_corners = -1, None
    for mask, sam_score in zip(masks, scores):
        mask = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        ok, corners = is_a4_contour(cnt, img_area)
        if ok:
            area_ratio  = cv2.contourArea(cnt) / img_area
            final_score = (0.30 * sam_score + 0.25 * rectangularity(cnt) +
                           0.15 * solidity(cnt) + 0.30 * area_ratio)
            if final_score > best_score:
                best_score, best_corners = final_score, corners

    del masks, scores
    free_memory()

    if best_corners is None:
        return None
    # Scale corners back to original image size
    corners_out = np.array(best_corners / scale, dtype=np.int32)
    if scale_input < 1.0:
        corners_out = (corners_out / scale_input).astype(np.int32)
    return corners_out


def detect_a4_fallback(image_bgr):
    """OpenCV fallback when SAM is unavailable."""
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        ok, corners = is_a4_contour(cnt, img_area)
        if ok:
            return np.array(corners, dtype=np.int32)
    return None


def compute_mm_per_px_from_a4(corners):
    """A4 = 210×297mm. corners = [tl, tr, br, bl]."""
    top_w    = np.linalg.norm(corners[1] - corners[0])
    bottom_w = np.linalg.norm(corners[2] - corners[3])
    px_w = (top_w + bottom_w) / 2.0
    left_h  = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])
    px_h = (left_h + right_h) / 2.0
    return ((210.0 / (px_w + 1e-6)) + (297.0 / (px_h + 1e-6))) / 2.0


# ══════════════════════════════════════════════════════════════
#  4.  UNet foot segmentation  (exact notebook + memory cleanup)
# ══════════════════════════════════════════════════════════════

def unet_segment_foot_fullres(image_bgr, model, device, img_size=256, thresh=0.5):
    model.eval()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = image_rgb.shape[:2]

    img_resized = cv2.resize(image_rgb, (img_size, img_size))
    del image_rgb
    img_norm = img_resized / 255.0
    del img_resized

    x = (torch.tensor(img_norm, dtype=torch.float32)
             .permute(2, 0, 1).unsqueeze(0).to(device))
    del img_norm

    with torch.no_grad():
        pred = model(x)[0, 0].detach().cpu().numpy()

    del x
    free_memory()

    mask_small = (pred > thresh).astype(np.uint8)
    del pred
    mask_full  = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
    del mask_small

    kernel    = np.ones((5, 5), np.uint8)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN,  kernel, iterations=1)
    free_memory()
    return mask_full


# ══════════════════════════════════════════════════════════════
#  5.  Foot length = MAX PERPENDICULAR DISTANCE from A4 bottom
#      (exact notebook methodology, vectorised)
# ══════════════════════════════════════════════════════════════

def foot_length_from_a4_bottom(mask, corners):
    bl = corners[3].astype(np.float32)
    br = corners[2].astype(np.float32)
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return None, None, bl, br
    pts   = np.stack([xs, ys], axis=1).astype(np.float32)
    AB    = br - bl
    AP    = pts - bl
    cross = np.abs(AB[0] * AP[:, 1] - AB[1] * AP[:, 0])
    dists = cross / (np.linalg.norm(AB) + 1e-6)
    idx   = np.argmax(dists)
    return float(dists[idx]), pts[idx], bl, br


# ══════════════════════════════════════════════════════════════
#  6.  Visualisation  (exact notebook colours/logic)
# ══════════════════════════════════════════════════════════════

def draw_results(image_bgr, corners, foot_mask, toe_point, bl, br, foot_len_cm, sizes):
    vis = image_bgr.copy()
    if corners is not None:
        cv2.polylines(vis, [corners.reshape(-1,1,2).astype(np.int32)], True, (0,255,0), 3)
    cv2.line(vis, tuple(bl.astype(int)), tuple(br.astype(int)), (255,0,255), 4)
    overlay = vis.copy()
    overlay[foot_mask == 1] = (0, 0, 255)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    del overlay
    if toe_point is not None:
        toe = tuple(toe_point.astype(int))
        cv2.circle(vis, toe, 10, (0,255,255), -1)
        AB  = br - bl
        t   = np.dot(toe_point - bl, AB) / (np.dot(AB, AB) + 1e-6)
        proj = tuple((bl + t * AB).astype(int))
        cv2.line(vis, toe, proj, (255,255,0), 4)
        cv2.circle(vis, proj, 8, (255,255,0), -1)
    text = f"{foot_len_cm:.2f} cm | UK {sizes['uk']} | US {sizes['us_men']} | EU {sizes['eu']}"
    cv2.putText(vis, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
    cv2.putText(vis, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 1)
    return vis


# ══════════════════════════════════════════════════════════════
#  7.  Pipeline class
# ══════════════════════════════════════════════════════════════

class ShoeSizePipeline:
    """
    Memory-efficient pipeline for Streamlit Cloud (1GB RAM).

    KEY DESIGN: SAM and UNet are NEVER in memory at the same time.
    ┌─────────────────────────────────────────────┐
    │  predict() flow:                            │
    │  1. Load SAM  → detect A4 → UNLOAD SAM     │  ~400MB freed
    │  2. Load UNet → segment foot → UNLOAD UNet │  ~100MB freed
    │  3. Measure + size lookup + draw            │  lightweight
    └─────────────────────────────────────────────┘
    Both checkpoint paths are stored and models are loaded
    fresh per-request, then immediately deleted.
    """

    def __init__(self, sam_ckpt_path=None, unet_ckpt_path=None):
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_ckpt_path  = sam_ckpt_path   # store path only — don't load yet
        self.unet_ckpt_path = unet_ckpt_path  # store path only — don't load yet

        # Verify both files exist at startup
        if sam_ckpt_path and os.path.exists(sam_ckpt_path):
            print(f"✅ SAM checkpoint found ({os.path.getsize(sam_ckpt_path)//1024//1024} MB)")
        else:
            print("⚠️  SAM checkpoint not found — will use OpenCV fallback")

        if unet_ckpt_path and os.path.exists(unet_ckpt_path):
            print(f"✅ UNet checkpoint found ({os.path.getsize(unet_ckpt_path)//1024//1024} MB)")
        else:
            print("⚠️  UNet checkpoint not found")

    # ── SAM: load → use → unload ──────────────────────────────
    def _load_sam(self):
        from segment_anything import sam_model_registry, SamPredictor
        print("[memory] Loading SAM into RAM…")
        sam = sam_model_registry["vit_b"](checkpoint=self.sam_ckpt_path)
        sam.to(self.device)
        predictor = SamPredictor(sam)
        print("[memory] SAM loaded ✅")
        return predictor

    def _unload_sam(self, predictor):
        try:
            del predictor.model
        except Exception:
            pass
        del predictor
        free_memory()
        print("[memory] SAM unloaded — RAM freed ✅")

    # ── UNet: load → use → unload ────────────────────────────
    def _load_unet(self):
        print("[memory] Loading UNet into RAM…")
        model = UNet().to(self.device)
        model.load_state_dict(
            torch.load(self.unet_ckpt_path, map_location=self.device,
                       weights_only=True)
        )
        model.eval()
        print("[memory] UNet loaded ✅")
        return model

    def _unload_unet(self, model):
        del model
        free_memory()
        print("[memory] UNet unloaded — RAM freed ✅")

    # ── Convenience properties (for status checks in app.py) ─
    @property
    def sam_predictor(self):
        """Returns True-ish if SAM checkpoint exists — used for status display only."""
        return self.sam_ckpt_path and os.path.exists(self.sam_ckpt_path)

    @property
    def unet_model(self):
        """Returns True-ish if UNet checkpoint exists — used for status display only."""
        return self.unet_ckpt_path and os.path.exists(self.unet_ckpt_path)

    # ── Main predict ─────────────────────────────────────────
    def predict(self, image_bgr, mode="ceil", jack_purcell=False, unet_thresh=0.5):
        result = {"ok": False}

        # Cap input resolution to save memory
        h0, w0 = image_bgr.shape[:2]
        if max(h0, w0) > 1920:
            scale = 1920 / max(h0, w0)
            image_bgr = cv2.resize(image_bgr,
                                   (int(w0*scale), int(h0*scale)),
                                   interpolation=cv2.INTER_AREA)
            print(f"[predict] Resized {w0}×{h0} → {image_bgr.shape[1]}×{image_bgr.shape[0]}")

        # ── PHASE 1: SAM in memory, UNet NOT loaded ──────────
        print("[predict] Phase 1: A4 detection (SAM)")
        if self.sam_ckpt_path and os.path.exists(self.sam_ckpt_path):
            predictor = self._load_sam()
            corners   = detect_a4_with_sam(image_bgr, predictor)
            self._unload_sam(predictor)   # ← SAM gone from RAM before UNet loads
        else:
            print("[predict] SAM not available — using OpenCV fallback")
            corners = detect_a4_fallback(image_bgr)
            free_memory()

        if corners is None:
            result["error"] = "❌ A4 NOT detected. Ensure the A4 sheet is fully visible, flat, and well-lit."
            return result

        corners_f = corners.astype(np.float32)
        mm_per_px = compute_mm_per_px_from_a4(corners_f)
        print(f"[predict] A4 detected ✅  mm/px = {mm_per_px:.5f}")

        # ── PHASE 2: UNet in memory, SAM already gone ────────
        print("[predict] Phase 2: Foot segmentation (UNet)")
        if not (self.unet_ckpt_path and os.path.exists(self.unet_ckpt_path)):
            result["error"] = "❌ UNet model not found."
            return result

        unet    = self._load_unet()
        foot_mask = unet_segment_foot_fullres(
            image_bgr, unet, self.device, thresh=unet_thresh
        )
        self._unload_unet(unet)   # ← UNet gone from RAM after segmentation

        # ── PHASE 3: Pure numpy/cv2 — no models needed ───────
        print("[predict] Phase 3: Measuring + sizing")
        foot_len_px, toe_point, bl, br = foot_length_from_a4_bottom(foot_mask, corners_f)

        if foot_len_px is None:
            result["error"] = "❌ Foot not detected in the image."
            return result

        foot_len_mm = foot_len_px * mm_per_px
        foot_len_cm = foot_len_mm / 10.0
        print(f"[predict] Foot length: {foot_len_cm:.2f} cm")

        sizes_raw = shoe_size_from_foot_length_cm(foot_len_cm, mode="floor", jack_purcell=jack_purcell)
        sizes_rec = shoe_size_from_foot_length_cm(foot_len_cm, mode=mode,    jack_purcell=jack_purcell)
        print(f"[predict] Size: UK {sizes_rec['uk']} | US {sizes_rec['us_men']} | EU {sizes_rec['eu']}")

        vis = draw_results(image_bgr, corners, foot_mask, toe_point, bl, br, foot_len_cm, sizes_rec)
        free_memory()

        result.update({
            "ok":         True,
            "length_cm":  round(foot_len_cm, 2),
            "length_mm":  round(foot_len_mm, 1),
            "mm_per_px":  round(mm_per_px, 6),
            "raw_size":   sizes_raw,
            "final_size": sizes_rec,
            "vis":        vis,
            "foot_mask":  foot_mask,
            "a4_corners": corners,
            "toe_point":  toe_point,
            "bl":         bl,
            "br":         br,
        })
        return result
