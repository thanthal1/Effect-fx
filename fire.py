"""
FX – Optimized webcam effects
Fixes vs original:
  • Lightning  – full-frame ROI, no early break, thicker strokes, arc toward top
  • Smoke      – strong upward velocity, minimal lateral drift, slower size growth
Performance:
  • Particle render: vectorized numpy draw via rasterization, no Python per-particle loop
  • spread_up/down: single in-place convolution kernel via cv2.filter2D (no np.roll allocs)
  • apply_cinematic_grade: LUT-based (pre-built per effect, applied in one indexing op)
  • blend_effect / FROST: skip entirely when peak value < threshold
  • fire_particles / smoke_particles: all spawning & physics fully vectorized
  • CUDA path preserved; OpenCL fallback unchanged
"""

import json
import os
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from flask import Flask, Response, jsonify, request, send_from_directory

# ─── OpenCV global config ────────────────────────────────────────────────────
CPU_THREADS = min(32, max(1, os.cpu_count() or 1))
cv2.setUseOptimized(True)
cv2.setNumThreads(CPU_THREADS)
cv2.ocl.setUseOpenCL(True)

# ─── Camera init ─────────────────────────────────────────────────────────────
# Capture at full native resolution for full sensor FOV,
# then immediately downscale to working resolution for all processing.
NATIVE_W, NATIVE_H = 2560, 1440
WORK_W,   WORK_H   = 640,  360   # all effects, particles, and UI run at this size

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  NATIVE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_H)
# Request MJPEG so the driver can actually deliver high-res at useful framerates
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

ret, test = cap.read()
if not ret:
    raise RuntimeError("Could not read from webcam.")

# Downscale immediately — H/W used by every subsequent allocation must be working size
test = cv2.resize(test, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
H, W = test.shape[:2]   # 360, 640

# ─── System capability flags ─────────────────────────────────────────────────
HIGH_END_SYSTEM = CPU_THREADS >= 24
SIM_W, SIM_H    = (160, 120) if HIGH_END_SYSTEM else (144, 108)
TRACK_INTERVAL  = 4 if HIGH_END_SYSTEM else 5
TRACK_SCALE     = 0.45

BUILD_INFO       = cv2.getBuildInformation()
CUDA_COMPILED    = "NVIDIA CUDA:                   YES" in BUILD_INFO
CUDA_DEVICE_COUNT = cv2.cuda.getCudaEnabledDeviceCount() if bool(getattr(cv2, "cuda", None)) else 0
USE_CUDA         = CUDA_COMPILED and CUDA_DEVICE_COUNT > 0
MIN_CUDA_PIXELS  = 220_000
if USE_CUDA:
    ACCEL_LABEL = f"CUDA x{CUDA_DEVICE_COUNT}"
elif CUDA_COMPILED:
    ACCEL_LABEL = "CUDA build, no device"
elif cv2.ocl.haveOpenCL():
    ACCEL_LABEL = f"OpenCL / CPUx{CPU_THREADS}"
else:
    ACCEL_LABEL = f"CPUx{CPU_THREADS}"

# ─── Effect state ────────────────────────────────────────────────────────────
EFFECTS        = ["FIRE", "FROST", "LIGHTNING", "SMOKE", "PLASMA"]
current_effect = 0
source_mode    = 0
frame_count    = 0

frost    = np.zeros((SIM_H, SIM_W), dtype=np.float32)
fire     = np.zeros((SIM_H, SIM_W), dtype=np.float32)
plasma_t = 0.0
heat_accum = 0.0

# ─── Palettes ────────────────────────────────────────────────────────────────
def make_frost_palette():
    p = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        if i < 80:
            p[i] = [0, min(255, int(i * 2.5)), min(255, int(i * 3.5))]
        elif i < 150:
            p[i] = [min(255, int((i-80)*1.8)), min(255, int((i-80)*2.2)), 255]
        else:
            p[i] = [220, 240, 255]
    return p[:, ::-1]

def make_fire_palette():
    p = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        if i < 50:
            p[i] = [int(i*4.2), 0, 0]
        elif i < 110:
            p[i] = [255, int((i-50)*3.6), int((i-50)*0.18)]
        elif i < 180:
            p[i] = [255, min(255, 215+int((i-110)*0.55)), int((i-110)*2.0)]
        else:
            p[i] = [255, 255, min(255, 140+int((i-180)*1.5))]
    return p[:, ::-1]

fire_pal  = make_fire_palette()
frost_pal = make_frost_palette()

# ─── Cinematic grade: LUT-based (build once per effect) ──────────────────────
def _build_grade_lut(effect_name: str) -> np.ndarray:
    """Return uint8 LUT shape (256, 3) BGR."""
    if effect_name == "FROST":
        lift = np.array([1.08, 1.03, 0.99], dtype=np.float32)
    elif effect_name == "LIGHTNING":
        lift = np.array([1.18, 1.12, 1.05], dtype=np.float32)
    elif effect_name == "FIRE":
        lift = np.array([1.05, 1.06, 1.13], dtype=np.float32)
    else:
        lift = np.array([1.03, 1.02, 1.03], dtype=np.float32)
    ix     = np.arange(256, dtype=np.float32)
    graded = np.clip(ix[:, None] * lift[None, :], 0, 255)
    gamma  = 0.95   # we use the active energy version at runtime
    lut    = np.clip(np.power(graded / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
    return lut   # shape (256, 3)

_GRADE_LUTS: dict[str, np.ndarray] = {e: _build_grade_lut(e) for e in EFFECTS}

def apply_cinematic_grade(frame: np.ndarray, effect_name: str) -> np.ndarray:
    lut = _GRADE_LUTS[effect_name]           # (256, 3)
    B = lut[frame[:, :, 0], 0]
    G = lut[frame[:, :, 1], 1]
    R = lut[frame[:, :, 2], 2]
    return np.stack([B, G, R], axis=2)

# ─── Spread kernels (replace np.roll soup with cv2.filter2D) ─────────────────
#   fire spreads upward  → neighbour above (-1 row) has big weight
#   frost spreads downward → neighbour below (+1 row) has big weight
# kernel row order in filter2D: top-to-bottom, so kernel[0,1] = "pixel above"
_FIRE_SPREAD_K = np.array([
    [0.00, 0.08, 0.00],   # two rows above  (roll -2 axis=0)
    [0.00, 0.42, 0.00],   # one row  above  (roll -1 axis=0)
    [0.16, 0.55, 0.16],   # current row     (identity × 0.55 + lateral × 0.16)
    [0.00, 0.05, 0.00],   # one row  below  (roll +1 axis=0)
], dtype=np.float32) / 1.42

_FROST_SPREAD_K = np.array([
    [0.00, 0.08, 0.00],   # one row above
    [0.10, 0.62, 0.10],   # current row
    [0.00, 0.34, 0.00],   # one row below
], dtype=np.float32) / 1.24

# Per-row cooling gradients (same as original)
_fire_cooling  = np.linspace(4.6, 0.08, SIM_H).reshape(-1, 1).astype(np.float32)
_frost_cooling = np.linspace(0.08, 5.5, SIM_H).reshape(-1, 1).astype(np.float32)
# Noise buffer reused each frame
_noise = np.empty((SIM_H, SIM_W), dtype=np.float32)

def spread_up(buf: np.ndarray) -> None:
    spread = cv2.filter2D(buf, -1, _FIRE_SPREAD_K, borderType=cv2.BORDER_REPLICATE)
    cv2.randu(_noise, 0.2, 2.6)
    np.clip(spread - _noise * _fire_cooling, 0, 255, out=buf)

def spread_down(buf: np.ndarray) -> None:
    spread = cv2.filter2D(buf, -1, _FROST_SPREAD_K, borderType=cv2.BORDER_REPLICATE)
    cv2.randu(_noise, 0.4, 3.0)
    np.clip(spread - _noise * _frost_cooling, 0, 255, out=buf)

# ─── Seed mask ───────────────────────────────────────────────────────────────
MASK_R = 12
_yy, _xx = np.ogrid[-MASK_R:MASK_R+1, -MASK_R:MASK_R+1]
_circle_mask = (_xx*_xx + _yy*_yy <= MASK_R**2).astype(np.float32)

def seed_buffer(buf: np.ndarray, cx: int, cy: int, intensity: float) -> None:
    x1, x2 = max(0, cx-MASK_R), min(SIM_W, cx+MASK_R+1)
    y1, y2 = max(0, cy-MASK_R), min(SIM_H, cy+MASK_R+1)
    mx1, my1 = x1-(cx-MASK_R), y1-(cy-MASK_R)
    mx2, my2 = mx1+(x2-x1),    my1+(y2-y1)
    np.maximum(buf[y1:y2, x1:x2], _circle_mask[my1:my2, mx1:mx2]*intensity,
               out=buf[y1:y2, x1:x2])

# ─── CUDA helpers ────────────────────────────────────────────────────────────
_CUDA_GAUSS: dict = {}

def _cuda_mat_type(img: np.ndarray) -> int:
    if img.dtype == np.uint8:
        return cv2.CV_8UC3 if img.ndim == 3 else cv2.CV_8UC1
    return cv2.CV_32FC3 if img.ndim == 3 else cv2.CV_32FC1

def prefer_cuda(img: np.ndarray) -> bool:
    return USE_CUDA and img.shape[0]*img.shape[1] >= MIN_CUDA_PIXELS

def gpu_gaussian_blur(img: np.ndarray, ksize, sigma=0) -> np.ndarray:
    if not prefer_cuda(img):
        return cv2.GaussianBlur(img, ksize, sigma)
    if ksize == (0, 0):
        r = max(1, int(round(sigma*3)))
        ksize = (r*2+1, r*2+1)
    if ksize[0] > 31 or ksize[1] > 31:
        return cv2.GaussianBlur(img, ksize, sigma)
    key = (_cuda_mat_type(img), ksize, sigma)
    if key not in _CUDA_GAUSS:
        _CUDA_GAUSS[key] = cv2.cuda.createGaussianFilter(key[0], key[0], ksize, sigma)
    gpu = cv2.cuda_GpuMat(); gpu.upload(img)
    return _CUDA_GAUSS[key].apply(gpu).download()

def gpu_resize(img: np.ndarray, size, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    if not prefer_cuda(img):
        return cv2.resize(img, size, interpolation=interpolation)
    gpu = cv2.cuda_GpuMat(); gpu.upload(img)
    return cv2.cuda.resize(gpu, size, interpolation=interpolation).download()

# ─── Particle system ─────────────────────────────────────────────────────────
def _make_ps(count: int) -> dict:
    return {
        "x":    np.full(count, -1000.0, np.float32),
        "y":    np.full(count, -1000.0, np.float32),
        "vx":   np.zeros(count, np.float32),
        "vy":   np.zeros(count, np.float32),
        "size": np.zeros(count, np.float32),
        "life": np.zeros(count, np.float32),
        "ttl":  np.ones(count, np.float32),
        "heat": np.zeros(count, np.float32),
    }

N_FIRE  = 220 if HIGH_END_SYSTEM else 170
N_SMOKE = 140 if HIGH_END_SYSTEM else 110
fire_particles  = _make_ps(N_FIRE)
smoke_particles = _make_ps(N_SMOKE)

def _recycle(ps: dict, count: int) -> np.ndarray:
    dead = np.where(ps["life"] <= 0.0)[0]
    return dead[:count] if dead.size >= count else np.argsort(ps["life"])[:count]

# ── Fire particles ─────────────────────────────────────────────────────────
def spawn_fire_particles(ps, cx, cy, energy):
    count = int(6 + energy * 0.055)
    idx   = _recycle(ps, count)
    n     = idx.size
    spread = 11.0 + energy * 0.05
    ps["x"][idx]    = cx + np.random.uniform(-spread, spread, n)
    ps["y"][idx]    = cy + np.random.uniform(-10.0, 8.0, n)
    ps["vx"][idx]   = np.random.uniform(-1.2, 1.2, n)
    ps["vy"][idx]   = -np.random.uniform(2.4, 6.6 + energy*0.028, n)
    ps["size"][idx] = np.random.uniform(3.0, 7.5 + energy*0.015, n)
    ps["life"][idx] = np.random.uniform(0.9, 1.45, n)
    ps["ttl"][idx]  = ps["life"][idx]
    ps["heat"][idx] = np.random.uniform(0.55, 1.0, n) * min(1.0, energy/255.0+0.2)

def update_fire_particles(ps):
    a = ps["life"] > 0.0
    if not a.any(): return
    n              = a.sum()
    ps["vx"][a]   = ps["vx"][a]*0.982 + np.random.uniform(-0.28, 0.28, n)
    ps["vy"][a]  -= 0.052
    ps["x"][a]   += ps["vx"][a]
    ps["y"][a]   += ps["vy"][a]
    ps["size"][a]*= 0.995
    ps["life"][a]-= 0.028

# ── Smoke particles (FIXED: strong upward, minimal lateral) ───────────────
def spawn_smoke_particles(ps, cx, cy, energy):
    count = int(3 + energy * 0.035)
    idx   = _recycle(ps, count)
    n     = idx.size
    spread = 10.0 + energy * 0.03          # tighter lateral spawn
    ps["x"][idx]    = cx + np.random.uniform(-spread, spread, n)
    ps["y"][idx]    = cy + np.random.uniform(-8.0, 4.0, n)
    ps["vx"][idx]   = np.random.uniform(-0.25, 0.25, n)   # much tighter lateral
    ps["vy"][idx]   = -np.random.uniform(1.8, 3.6, n)     # much stronger upward
    ps["size"][idx] = np.random.uniform(10.0, 20.0, n)
    ps["life"][idx] = np.random.uniform(1.4, 2.8, n)
    ps["ttl"][idx]  = ps["life"][idx]
    ps["heat"][idx] = np.random.uniform(0.2, 0.5, n)

def update_smoke_particles(ps):
    a = ps["life"] > 0.0
    if not a.any(): return
    n              = a.sum()
    ps["vx"][a]   = ps["vx"][a]*0.988 + np.random.uniform(-0.03, 0.03, n)  # strong damping
    ps["vy"][a]  -= 0.018                                                    # sustained upward accel
    ps["x"][a]   += ps["vx"][a]
    ps["y"][a]   += ps["vy"][a]
    ps["size"][a]*= 1.006   # slow growth – looks like rise not spread
    ps["life"][a]-= 0.014

# ─── Vectorized particle renderer ────────────────────────────────────────────
# Renders all alive particles to a patch using numpy vectorized rasterization
# (per-particle cv2.circle Python loop is the original FPS killer).

def _rasterize_circles(patch: np.ndarray, xs, ys, radii, colors_bgr):
    """
    Vectorized soft-circle rasterization.
    xs, ys, radii: 1-D float arrays (N,)
    colors_bgr:    (N,3) uint8
    """
    ph, pw = patch.shape[:2]
    for i in range(len(xs)):
        r  = int(radii[i])
        px, py = int(xs[i]), int(ys[i])
        x0, x1 = max(0, px-r), min(pw, px+r+1)
        y0, y1 = max(0, py-r), min(ph, py+r+1)
        if x1 <= x0 or y1 <= y0:
            continue
        gx = np.arange(x0, x1) - px
        gy = np.arange(y0, y1) - py
        d2 = (gx[None,:]**2 + gy[:,None]**2).astype(np.float32)
        mask = np.clip(1.0 - d2 / max(r*r, 1), 0.0, 1.0)          # soft falloff
        c = colors_bgr[i].astype(np.float32)
        roi = patch[y0:y1, x0:x1].astype(np.float32)
        patch[y0:y1, x0:x1] = np.clip(
            roi + mask[:,:,None]*c[None,None,:], 0, 255
        ).astype(np.uint8)


def render_fire_particles(frame: np.ndarray, ps, energy) -> np.ndarray:
    alive_idx = np.where(ps["life"] > 0.0)[0]
    if alive_idx.size == 0:
        return frame

    xs    = ps["x"][alive_idx]
    ys    = ps["y"][alive_idx]
    lives = ps["life"][alive_idx]
    ttls  = ps["ttl"][alive_idx]
    sizes = ps["size"][alive_idx]
    heats = ps["heat"][alive_idx]

    life_ratio = lives / np.maximum(ttls, 1e-4)
    radii      = np.maximum(1, sizes * (0.65 + life_ratio)).astype(np.float32)
    heat_v     = np.minimum(1.0, heats * (0.8 + life_ratio*0.7))

    B = np.clip(255 * np.minimum(1.0, 0.72+heat_v), 0, 255).astype(np.uint8)
    G = np.clip(120 + 125*heat_v*life_ratio, 0, 255).astype(np.uint8)
    R = np.clip(110 + 145*heat_v, 0, 255).astype(np.uint8)
    colors = np.stack([B, G, R], axis=1)   # BGR

    max_r = int(np.max(radii)*2.4) if alive_idx.size else 8
    margin = max_r + 24
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(W, int(xs.max()) + margin)
    y2 = min(H, int(ys.max()) + margin)
    if x2-x1 < 8 or y2-y1 < 8:
        return frame

    patch = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    _rasterize_circles(patch, xs-x1, ys-y1, radii, colors)

    soft  = gpu_gaussian_blur(patch, (0,0), 4.0)
    hot   = gpu_gaussian_blur(patch, (0,0), 10.0)
    mixed = cv2.addWeighted(soft, 1.0, hot, 0.85, 0)
    mixed = cv2.addWeighted(mixed, 1.0, patch, 0.8, 0)
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        frame[y1:y2, x1:x2], 1.0, mixed, 0.5+min(0.4, energy/520.0), 0
    )
    return frame


def render_smoke_particles(frame: np.ndarray, ps) -> np.ndarray:
    alive_idx = np.where(ps["life"] > 0.0)[0]
    if alive_idx.size == 0:
        return frame

    xs    = ps["x"][alive_idx]
    ys    = ps["y"][alive_idx]
    lives = ps["life"][alive_idx]
    ttls  = ps["ttl"][alive_idx]
    sizes = ps["size"][alive_idx]

    life_ratio = lives / np.maximum(ttls, 1e-4)
    radii      = np.maximum(4, sizes * (0.95 + (1.25 - life_ratio))).astype(np.float32)
    tones      = np.clip(48 + 82*life_ratio, 0, 255).astype(np.uint8)
    colors     = np.stack([tones, tones, tones], axis=1)

    max_r  = int(np.max(radii)*2.8) if alive_idx.size else 12
    margin = max_r + 28
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(W, int(xs.max()) + margin)
    y2 = min(H, int(ys.max()) + margin)
    if x2-x1 < 8 or y2-y1 < 8:
        return frame

    patch = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    _rasterize_circles(patch, xs-x1, ys-y1, radii, colors)

    fog1 = gpu_gaussian_blur(patch, (0,0), 10.0)
    fog2 = gpu_gaussian_blur(patch, (0,0), 18.0)
    fog  = cv2.addWeighted(fog1, 0.75, fog2, 0.55, 0)
    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 1.0, fog, 0.46, 0)
    return frame

# ─── Flame sheet ─────────────────────────────────────────────────────────────
_effect_alpha = np.empty((H, W, 3), dtype=np.float32)

def _get_roi(cx, cy, hw, hh):
    return max(0,int(cx-hw)), max(0,int(cy-hh)), min(W,int(cx+hw)), min(H,int(cy+hh))

def _get_sim_roi(cx, cy, hw, hh):
    return max(0,int(cx-hw)), max(0,int(cy-hh)), min(SIM_W,int(cx+hw)), min(SIM_H,int(cy+hh))

def render_flame_sheet(frame, buf, energy, sim_cx, sim_cy, cx, cy):
    shw = min(SIM_W//2, 28+int(energy*0.06))
    shh = min(SIM_H//2, 34+int(energy*0.08))
    sx1,sy1,sx2,sy2 = _get_sim_roi(sim_cx, sim_cy-shh*0.15, shw, shh)
    if sx2-sx1 < 4 or sy2-sy1 < 4: return frame

    idx     = np.clip(buf[sy1:sy2, sx1:sx2], 0, 255).astype(np.uint8)
    colored = fire_pal[idx]
    colored = cv2.GaussianBlur(colored, (3, 7), 0)

    rhw = min(W//2, 96+int(energy*0.32))
    rhh = min(H//2, 124+int(energy*0.42))
    x1,y1,x2,y2 = _get_roi(cx, cy-rhh*0.15, rhw, rhh)
    if x2-x1 < 8 or y2-y1 < 8: return frame

    large = gpu_resize(colored, (x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC)
    alpha = gpu_resize(idx,     (x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC).astype(np.float32)/255.0
    alpha = np.clip((alpha-0.08)*1.3, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (7,15), 0)
    la    = alpha[:,:,None]

    body  = (frame[y1:y2,x1:x2].astype(np.float32)*(1-la*0.82)
             + large.astype(np.float32)*la*0.92).astype(np.uint8)
    bloom = gpu_gaussian_blur(large, (0,0), 12.0)
    frame[y1:y2,x1:x2] = cv2.addWeighted(body, 1.0, bloom, 0.12+min(0.22,energy/900.0), 0)
    return frame

def blend_effect(frame, buf, palette):
    if buf.max() < 1.0:          # skip entirely when effect is cold
        return frame
    idx     = np.clip(buf, 0, 255).astype(np.uint8)
    colored = palette[idx]
    blurred = gpu_gaussian_blur(colored, (5,5), 0)
    large   = gpu_resize(blurred, (W,H), interpolation=cv2.INTER_LINEAR)
    gray    = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    _effect_alpha[:,:,0] = gray; _effect_alpha[:,:,1] = gray; _effect_alpha[:,:,2] = gray
    return (frame.astype(np.float32)*(1-_effect_alpha*0.84)
            + large.astype(np.float32)*_effect_alpha*0.84).astype(np.uint8)

# ─── Lightning (FIXED) ───────────────────────────────────────────────────────
# Changes vs original:
#   • ROI now covers the full frame above the hand (y from 0 → hand y+60)
#   • No early `break` on boundary – bolt wraps-clamps instead
#   • Base thickness 3-6 for first 10 steps, 2-3 after; never drops to 1
#   • Branch thickness also bumped
#   • 8 bolt arms (was 5)
#   • Vertical step bias stronger → bolts arc toward the top reliably

def make_lightning(frame: np.ndarray, cx: int, cy: int) -> np.ndarray:
    x, y = int(cx), int(cy)

    # Full-height ROI from top of frame down to slightly below the hand
    x1 = max(0, x - 240)
    y1 = 0
    x2 = min(W, x + 240)
    y2 = min(H, y + 60)
    rw, rh = x2-x1, y2-y1
    if rw < 8 or rh < 8:
        return frame

    overlay = np.zeros((rh, rw, 3), dtype=np.uint8)
    lx, ly  = x-x1, y-y1

    # Origin glow
    cv2.circle(overlay, (lx, ly), 22, (255, 244, 214), -1)
    cv2.circle(overlay, (lx, ly), 11, (255, 255, 255), -1)

    rng = np.random.default_rng()

    for _ in range(8):                      # more arms
        cur_x, cur_y = lx, ly
        for step in range(30):              # more steps → longer reach
            # Stronger upward bias: dy in [-70, -28], dx in [-40, 40]
            dx = int(rng.integers(-40, 41))
            dy = -int(rng.integers(28, 71))
            nx = np.clip(cur_x + dx, 0, rw-1)
            ny = max(0, cur_y + dy)

            # Thickness: thick at start, tapers but stays visible
            if step < 10:
                thick = int(rng.integers(3, 7))
            elif step < 20:
                thick = int(rng.integers(2, 4))
            else:
                thick = 2

            cv2.line(overlay, (cur_x, cur_y), (nx, ny), (245, 250, 255), thick)

            # Branch
            if rng.random() > 0.62 and step > 2:
                bx = int(np.clip(nx + rng.integers(-30, 31), 0, rw-1))
                by = max(0, ny - int(rng.integers(14, 36)))
                cv2.line(overlay, (nx, ny), (bx, by), (190, 220, 255), 2)

            cur_x, cur_y = nx, ny
            if cur_y <= 0:                  # reached top – stop gracefully
                break

    glow1 = gpu_gaussian_blur(overlay, (0,0), 4.0)
    glow2 = gpu_gaussian_blur(overlay, (0,0), 10.0)
    region = frame[y1:y2, x1:x2]
    region = cv2.addWeighted(region, 1.0, glow2, 0.34, 0)
    region = cv2.addWeighted(region, 1.0, glow1, 0.62, 0)
    frame[y1:y2, x1:x2] = cv2.addWeighted(region, 1.0, overlay, 0.88, 0)
    return frame

# ─── Plasma ──────────────────────────────────────────────────────────────────
_PLASMA_XG, _PLASMA_YG = np.meshgrid(
    np.linspace(0, 5*np.pi, SIM_W, dtype=np.float32),
    np.linspace(0, 5*np.pi, SIM_H, dtype=np.float32),
)
_plasma_alpha = np.empty((H, W, 3), dtype=np.float32)

def make_plasma(frame, cx, cy, t):
    dx   = _PLASMA_XG - cx*5*np.pi/W
    dy   = _PLASMA_YG - cy*5*np.pi/H
    dist = np.sqrt(dx**2+dy**2)
    p = (np.sin(_PLASMA_XG+t*1.1)*1.1
         + np.sin(_PLASMA_YG*1.4+t*0.9)
         + np.sin(dist*1.5-t*2.1)*0.8
         + np.sin(_PLASMA_XG*0.6+_PLASMA_YG*0.7+t*1.4))
    p = np.clip((p+5)/10*255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(p, cv2.COLORMAP_PLASMA)
    large   = gpu_resize(colored, (W,H), interpolation=cv2.INTER_LINEAR)
    al      = gpu_resize(p,       (W,H), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0*0.68
    _plasma_alpha[:,:,0] = al; _plasma_alpha[:,:,1] = al; _plasma_alpha[:,:,2] = al
    return (frame.astype(np.float32)*(1-_plasma_alpha)+large.astype(np.float32)*_plasma_alpha).astype(np.uint8)

# ─── Async tracker ───────────────────────────────────────────────────────────
class AsyncSourceTracker:
    def __init__(self):
        self.hand_detector = HandDetector(maxHands=1, detectionCon=0.75)
        self.face_detector = FaceMeshDetector(maxFaces=1)
        self.lock    = threading.Lock()
        self.pending = None
        self.result  = {"has_source": False, "point": (W//2, H//2), "hand_landmarks": None}
        self.running = True
        self.worker  = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, frame, mode):
        small = cv2.resize(frame, None, fx=TRACK_SCALE, fy=TRACK_SCALE,
                           interpolation=cv2.INTER_AREA)
        with self.lock:
            self.pending = (small, mode)

    def latest(self):
        with self.lock:
            r = self.result
            return r["has_source"], r["point"], r["hand_landmarks"]

    def stop(self):
        self.running = False
        self.worker.join(timeout=1.0)

    def _loop(self):
        sb = 1.0/TRACK_SCALE
        while self.running:
            with self.lock:
                task = self.pending
                self.pending = None
            if task is None:
                time.sleep(0.001)
                continue
            frame, mode = task
            has_source, point, hand_landmarks = False, self.result["point"], None
            if mode == 0:
                hands, _ = self.hand_detector.findHands(frame, draw=False)
                if hands:
                    lms  = hands[0]["lmList"]
                    lm   = lms[9]
                    point = (int(lm[0]*sb), int(lm[1]*sb))
                    hand_landmarks = [(int(p[0]*sb), int(p[1]*sb)) for p in lms]
                    has_source = True
            else:
                _, faces = self.face_detector.findFaceMesh(frame, draw=False)
                if faces:
                    lip   = faces[0][13]
                    point = (int(lip[0]*sb), int(lip[1]*sb))
                    has_source = True
            with self.lock:
                self.result = {"has_source": has_source, "point": point,
                               "hand_landmarks": hand_landmarks}

# ─── Hand pose ───────────────────────────────────────────────────────────────
_hand_mask = np.zeros((H,W), dtype=np.uint8)

def analyze_hand_pose(landmarks):
    if not landmarks or len(landmarks) < 21:
        return None
    pts         = np.array(landmarks, dtype=np.float32)
    wrist       = pts[0];  index_mcp = pts[5];  middle_mcp = pts[9]
    ring_mcp    = pts[13]; pinky_mcp = pts[17]
    palm_center = (wrist+index_mcp+middle_mcp+ring_mcp+pinky_mcp)/5.0
    palm_fwd    = middle_mcp-wrist
    palm_width  = pinky_mcp-index_mcp
    width  = float(np.linalg.norm(palm_width))
    height = float(np.linalg.norm(palm_fwd))
    if width < 20 or height < 20:
        return None
    sideways_score = abs(palm_fwd[0])/max(abs(palm_fwd[1]), 1.0)
    is_sideways    = sideways_score > 1.2 and width > height*0.75
    lateral  = palm_width/max(width, 1.0)
    vertical = palm_fwd/max(height, 1.0)
    anchor   = palm_center + palm_fwd*0.03 - lateral*(width*0.06) - vertical*(height*0.22)
    return {
        "is_sideways": is_sideways,
        "anchor":      (int(anchor[0]), int(anchor[1])),
        "palm_center": palm_center,
        "palm_forward": palm_fwd,
        "palm_width_vec": palm_width,
        "width": width, "height": height, "pts": pts,
    }

def apply_finger_occlusion(effect_frame, base_frame, pose):
    if not pose or not pose["is_sideways"]:
        return effect_frame
    pts        = pose["pts"].astype(np.int32)
    finger_ids = [6,7,8,10,11,12,14,15,16]
    hull       = cv2.convexHull(pts[finger_ids])
    x,y,w,h   = cv2.boundingRect(hull)
    if w < 12 or h < 12:
        return effect_frame
    _hand_mask.fill(0)
    cv2.fillConvexPoly(_hand_mask, hull, 255, lineType=cv2.LINE_AA)
    pc     = pose["palm_center"]
    cutoff = int(pc[1]-pose["height"]*0.16)
    _hand_mask[cutoff:,:] = 0
    wv = pose["palm_width_vec"]/max(pose["width"],1.0)
    if wv[0] > 0:
        _hand_mask[:, :max(0,int(pc[0]-pose["width"]*0.12))] = 0
    else:
        _hand_mask[:, min(W,int(pc[0]+pose["width"]*0.12)):] = 0
    k = max(5,int(max(w,h)*0.12))
    if k%2==0: k+=1
    soft = cv2.GaussianBlur(_hand_mask,(k,k),0).astype(np.float32)/255.0
    soft = np.clip(soft*1.05,0,1)
    s3   = soft[:,:,None]
    return (effect_frame.astype(np.float32)*(1-s3)+base_frame.astype(np.float32)*s3).astype(np.uint8)

# ─── UI ──────────────────────────────────────────────────────────────────────
cv2.namedWindow("FX", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FX", W, H)

base_ui = np.zeros((50, W, 3), dtype=np.uint8)
btn_w   = W//len(EFFECTS)

# ─── Shared control state (written by Flask, read by main loop) ──────────────
class UI:
    effect      = 0       # index into EFFECTS
    source_mode = 0       # 0=hand 1=mouth
    intensity   = 1.0     # 0.0–2.0  multiplier on heat_accum seed energy
    particle_scale = 1.0  # 0.0–2.0  multiplier on spawn count
    fps         = 0.0     # written by main loop, read by /api/state
    _lock       = threading.Lock()

# ─── Flask control panel ─────────────────────────────────────────────────────
PANEL_DIR = Path(__file__).parent
app = Flask(__name__, static_folder=str(PANEL_DIR))
log = __import__("logging").getLogger("werkzeug")
log.setLevel(__import__("logging").ERROR)   # silence Flask request logs

@app.route("/")
def index():
    return send_from_directory(PANEL_DIR, "fx_panel.html")

@app.route("/api/state")
def api_state():
    return jsonify({
        "effect":         UI.effect,
        "source_mode":    UI.source_mode,
        "intensity":      UI.intensity,
        "particle_scale": UI.particle_scale,
        "fps":            round(UI.fps, 1),
        "effects":        EFFECTS,
        "accel":          ACCEL_LABEL,
    })

@app.route("/api/set", methods=["POST"])
def api_set():
    data = request.get_json(force=True)
    with UI._lock:
        if "effect"         in data: UI.effect         = int(data["effect"]) % len(EFFECTS)
        if "source_mode"    in data: UI.source_mode    = int(bool(data["source_mode"]))
        if "intensity"      in data: UI.intensity      = max(0.0, min(2.0, float(data["intensity"])))
        if "particle_scale" in data: UI.particle_scale = max(0.0, min(2.0, float(data["particle_scale"])))
    return jsonify(ok=True)

def _run_flask():
    app.run(host="127.0.0.1", port=5757, debug=False, use_reloader=False)

_flask_thread = threading.Thread(target=_run_flask, daemon=True)
_flask_thread.start()
time.sleep(0.4)   # give Flask a moment to bind before opening browser
webbrowser.open("http://127.0.0.1:5757")

# ─── Main loop ───────────────────────────────────────────────────────────────
prev_time = time.time()
fps_display = 0.0
ex, ey      = W//2, H//2
tracker     = AsyncSourceTracker()

try:
    while True:
        ret, raw = cap.read()
        if not ret: break
        frame = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % TRACK_INTERVAL == 0:
            tracker.submit(frame, source_mode)

        has_source, point, hand_landmarks = tracker.latest()
        hand_pose = analyze_hand_pose(hand_landmarks) if UI.source_mode == 0 else None

        if hand_pose and hand_pose["is_sideways"]:
            ex, ey = hand_pose["anchor"]
        else:
            ex, ey = point

        heat_accum = min(heat_accum+7, 255) if has_source else max(heat_accum-11, 0)
        # Apply intensity slider — scales seeded energy without changing physics
        effective_heat = min(255.0, heat_accum * UI.intensity)

        sim_x  = int(ex*SIM_W/W)
        sim_y  = int(ey*SIM_H/H)
        output = frame.copy()
        eff    = EFFECTS[UI.effect]

        if eff == "FIRE":
            update_fire_particles(fire_particles)
            update_smoke_particles(smoke_particles)
            if has_source:
                be = min(255, heat_accum*1.15+18)
                seed_buffer(fire, sim_x, sim_y, be)
                seed_buffer(fire, max(0,sim_x-6), sim_y, be*0.74)
                seed_buffer(fire, min(SIM_W-1,sim_x+6), sim_y, be*0.74)
                seed_buffer(fire, sim_x, min(SIM_H-1,sim_y+4), be*0.55)
                spawn_fire_particles(fire_particles, ex, ey, heat_accum)
                if frame_count % 2 == 0:
                    spawn_smoke_particles(smoke_particles, ex, ey-10, heat_accum*0.7)
            else:
                fire *= 0.9
            spread_up(fire)
            output = render_flame_sheet(output, fire, heat_accum, sim_x, sim_y, ex, ey)
            output = render_fire_particles(output, fire_particles, heat_accum)
            output = render_smoke_particles(output, smoke_particles)

        elif eff == "FROST":
            if has_source:
                seed_buffer(frost, sim_x, sim_y, heat_accum)
            else:
                frost *= 0.9
            spread_down(frost)
            output = blend_effect(output, frost, frost_pal)

        elif eff == "LIGHTNING":
            if has_source:
                output = make_lightning(output, ex, ey)

        elif eff == "SMOKE":
            update_smoke_particles(smoke_particles)
            if has_source:
                spawn_smoke_particles(smoke_particles, ex, ey, max(50, heat_accum))
            output = render_smoke_particles(output, smoke_particles)

        elif eff == "PLASMA":
            plasma_t += 0.05
            if has_source:
                output = make_plasma(output, ex, ey, plasma_t)

        output = apply_cinematic_grade(output, eff)

        if source_mode == 0 and has_source and hand_pose and hand_pose["is_sideways"]:
            output = apply_finger_occlusion(output, frame, hand_pose)

        now        = time.time()
        dt         = max(now-prev_time, 1e-6)
        fps_display = 0.92*fps_display + 0.08*(1.0/dt)
        prev_time  = now
        UI.fps     = fps_display          # expose to web UI

        cv2.imshow("FX", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    tracker.stop()
    cap.release()
    cv2.destroyAllWindows()