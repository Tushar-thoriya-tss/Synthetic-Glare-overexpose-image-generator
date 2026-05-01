"""Glare / overexposure rendering engine.

Pure numpy/OpenCV. No UI dependencies. Builds a single-channel float32
"glare intensity map" per glare and composes it into the source image,
returning the synthesized image and a binary mask.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Per-shape renderers — each returns a HxW float32 map of *added* brightness  #
# --------------------------------------------------------------------------- #

def _meshgrid(h: int, w: int):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs, ys


def render_ellipse(h, w, cx, cy, rx, ry, angle_deg, peak, softness):
    xs, ys = _meshgrid(h, w)
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = xs - cx, ys - cy
    xp = cos_t * dx + sin_t * dy
    yp = -sin_t * dx + cos_t * dy
    nd2 = (xp / max(rx, 1.0)) ** 2 + (yp / max(ry, 1.0)) ** 2
    sigma2 = max(softness, 0.05) ** 2
    return (peak * np.exp(-nd2 / (2.0 * sigma2))).astype(np.float32)


def render_line(h, w, x1, y1, x2, y2, width, peak, softness):
    xs, ys = _meshgrid(h, w)
    vx, vy = x2 - x1, y2 - y1
    L = max(float(np.hypot(vx, vy)), 1.0)
    ux, uy = vx / L, vy / L
    px, py = xs - x1, ys - y1
    along = px * ux + py * uy
    perp = px * (-uy) + py * ux
    sigma_cross = max(width * softness, 0.5)
    cross = np.exp(-(perp ** 2) / (2.0 * sigma_cross ** 2))
    end_sigma = max(width, 1.0)
    end = np.where(along < 0, np.exp(-(along ** 2) / (2.0 * end_sigma ** 2)), 1.0)
    end = np.where(along > L, np.exp(-((along - L) ** 2) / (2.0 * end_sigma ** 2)), end)
    return (peak * cross * end).astype(np.float32)


def render_polygon(h, w, points, peak, softness, blur_sigma):
    mask = np.zeros((h, w), dtype=np.float32)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) >= 3:
        cv2.fillPoly(mask, [pts], 1.0)
    sigma = max(blur_sigma * softness, 0.5)
    k = int(max(sigma * 3, 3)) | 1
    mask = cv2.GaussianBlur(mask, (k, k), sigma)
    m = max(float(mask.max()), 1e-6)
    return (peak * mask / m).astype(np.float32)


def render_freeform(h, w, points, brush, peak, softness):
    mask = np.zeros((h, w), dtype=np.float32)
    pts = np.array(points, dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 1.0,
                      thickness=int(max(brush * 2, 1)))
    elif len(pts) == 1:
        cv2.circle(mask, tuple(pts[0]), int(max(brush, 1)), 1.0, -1)
    sigma = max(brush * softness, 0.5)
    k = int(max(sigma * 3, 3)) | 1
    mask = cv2.GaussianBlur(mask, (k, k), sigma)
    m = max(float(mask.max()), 1e-6)
    return (peak * mask / m).astype(np.float32)


def render_texture(h, w, texture_gray, cx, cy, tw, th, angle_deg, peak):
    if texture_gray is None or texture_gray.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    tex = cv2.resize(texture_gray, (max(int(tw), 1), max(int(th), 1)),
                     interpolation=cv2.INTER_LINEAR).astype(np.float32)
    tex /= max(float(tex.max()), 1e-6)
    th_, tw_ = tex.shape[:2]
    M = cv2.getRotationMatrix2D((tw_ / 2.0, th_ / 2.0), angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nW = int(th_ * sin_a + tw_ * cos_a)
    nH = int(th_ * cos_a + tw_ * sin_a)
    M[0, 2] += nW / 2.0 - tw_ / 2.0
    M[1, 2] += nH / 2.0 - th_ / 2.0
    tex_r = cv2.warpAffine(tex, M, (nW, nH), borderValue=0)

    out = np.zeros((h, w), dtype=np.float32)
    x0 = int(round(cx - nW / 2.0)); y0 = int(round(cy - nH / 2.0))
    x1, y1 = x0 + nW, y0 + nH
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x1), min(h, y1)
    if dx1 > dx0 and dy1 > dy0:
        out[dy0:dy1, dx0:dx1] = tex_r[sy0:sy0 + (dy1 - dy0),
                                      sx0:sx0 + (dx1 - dx0)]
    return (out * peak).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dispatch + composition                                                      #
# --------------------------------------------------------------------------- #

def render_glare(glare: dict, h: int, w: int, textures: dict | None = None) -> np.ndarray:
    t = glare["type"]
    p = dict(glare["params"])  # copy so we don't mutate
    if t == "ellipse":
        return render_ellipse(h, w, **p)
    if t == "line":
        return render_line(h, w, **p)
    if t == "polygon":
        return render_polygon(h, w, **p)
    if t == "freedraw":
        return render_freeform(h, w, **p)
    if t == "texture":
        name = p.pop("texture_name", None)
        tex = textures.get(name) if (textures and name) else None
        return render_texture(h, w, tex, **p)
    return np.zeros((h, w), dtype=np.float32)


def compose(image: np.ndarray,
            glares: list,
            mask_threshold: float = 30.0,
            tint=(1.0, 1.0, 1.0),
            noise_std: float = 0.0,
            textures: dict | None = None,
            blend: str = "add") -> tuple[np.ndarray, np.ndarray]:
    """Compose all glares onto the image.

    image          : HxWx3 uint8 (RGB)
    glares         : list of {"type": ..., "params": {...}}
    mask_threshold : intensity (0..255) above which a pixel is in the binary mask
    tint           : per-channel multiplier applied to the glare intensity
    noise_std      : sensor-style Gaussian noise scaled by glare strength
    blend          : "add" (clipped — true overexposure) or "screen" (softer)

    Returns (synthesized_uint8_RGB, binary_mask_uint8_0_or_255).
    """
    h, w = image.shape[:2]
    intensity = np.zeros((h, w), dtype=np.float32)
    for g in glares:
        intensity = np.maximum(intensity, render_glare(g, h, w, textures))

    img_f = image.astype(np.float32)
    glare_rgb = np.stack([intensity * tint[0],
                          intensity * tint[1],
                          intensity * tint[2]], axis=-1)

    if blend == "screen":
        b = np.clip(glare_rgb, 0, 255)
        out = 255.0 - (255.0 - img_f) * (255.0 - b) / 255.0
    else:
        out = img_f + glare_rgb

    if noise_std > 0:
        scale = (intensity / 255.0)[..., None]
        out = out + np.random.normal(0, noise_std, out.shape).astype(np.float32) * scale

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    mask = (intensity >= mask_threshold).astype(np.uint8) * 255
    return out_u8, mask


# --------------------------------------------------------------------------- #
# Random generator (auto mode)                                                #
# --------------------------------------------------------------------------- #

def _skewed_peak(rng: random.Random) -> float:
    """Bimodal peak distribution: ~70% faint, ~30% strong.

    Faint glares dominate so the model can't just learn "find white blobs",
    while a minority of strong cases keep severe-overexposure coverage.
    """
    if rng.random() < 0.70:
        return rng.uniform(40, 120)   # faint / mild wash
    return rng.uniform(150, 230)      # strong / clipped


def _maybe_halo_for(core: dict, rng: random.Random) -> dict | None:
    """Pair a bright elliptical core with a large soft halo (real-world bleed).

    Only fires for bright cores (peak ≥ 150) and only ~35% of the time, so
    faint glares stay isolated. Halo shares position/orientation, scaled
    2.5–4× larger, very low peak (25–55), very soft falloff.
    """
    if core["peak"] < 150 or rng.random() >= 0.35:
        return None
    scale = rng.uniform(2.5, 4.0)
    return dict(
        cx=core["cx"], cy=core["cy"],
        rx=core["rx"] * scale, ry=core["ry"] * scale,
        angle_deg=core["angle_deg"],
        peak=rng.uniform(25, 55),
        softness=rng.uniform(1.0, 1.5),
    )


def random_glares(h: int, w: int, n: int | None = None,
                  texture_names: Iterable[str] | None = None,
                  rng: random.Random | None = None) -> list:
    rng = rng or random
    if n is None:
        # Weighted toward 1 (most realistic), with rare 0/3 and frequent 2.
        # Used only when caller doesn't pass n (UI passes it explicitly).
        n = rng.choices([0, 1, 2, 3], weights=[5, 55, 30, 10])[0]
    types = ["ellipse", "line", "polygon"]
    if texture_names:
        types.append("texture")

    short_side = min(h, w)
    out = []
    for _ in range(n):
        t = rng.choice(types)
        if t == "ellipse":
            core = dict(
                cx=rng.randint(0, w),                      # allow edge clipping
                cy=rng.randint(0, h),
                rx=rng.randint(int(w * 0.05), int(w * 0.20)),
                ry=rng.randint(int(h * 0.05), int(h * 0.20)),
                angle_deg=rng.uniform(0, 180),
                peak=_skewed_peak(rng),
                softness=rng.uniform(0.5, 1.2),            # wider falloff range
            )
            halo = _maybe_halo_for(core, rng)
            if halo is not None:
                # Halo first so core paints on top via np.maximum in compose.
                out.append({"type": "ellipse", "params": halo})
            out.append({"type": "ellipse", "params": core})
        elif t == "line":
            x1 = rng.randint(0, w); y1 = rng.randint(0, h)
            ang = rng.uniform(0, np.pi)
            length = rng.randint(int(w * 0.2), int(w * 0.6))
            out.append({"type": "line", "params": dict(
                x1=x1, y1=y1,
                x2=int(x1 + np.cos(ang) * length),
                y2=int(y1 + np.sin(ang) * length),
                # image-size-aware streak width — ~0.5–2% of short side
                width=rng.uniform(short_side * 0.005, short_side * 0.020),
                peak=_skewed_peak(rng),
                softness=rng.uniform(0.5, 1.0),
            )})
        elif t == "polygon":
            cx = rng.randint(int(w * 0.05), int(w * 0.95))
            cy = rng.randint(int(h * 0.05), int(h * 0.95))
            r = rng.randint(int(short_side * 0.04), int(short_side * 0.18))
            n_pts = rng.randint(5, 9)
            pts = []
            for i in range(n_pts):
                a = 2 * np.pi * i / n_pts + rng.uniform(-0.3, 0.3)
                rr = r * rng.uniform(0.6, 1.2)
                pts.append([int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))])
            out.append({"type": "polygon", "params": dict(
                points=pts,
                peak=_skewed_peak(rng),
                softness=1.0,
                # blur scales with polygon size — small shapes stay sharp,
                # large shapes get proportionally softer edges
                blur_sigma=r * rng.uniform(0.4, 0.8),
            )})
        else:  # texture
            name = rng.choice(list(texture_names))
            out.append({"type": "texture", "params": dict(
                texture_name=name,
                cx=rng.randint(0, w),
                cy=rng.randint(0, h),
                tw=rng.randint(int(w * 0.10), int(w * 0.40)),
                th=rng.randint(int(h * 0.10), int(h * 0.40)),
                angle_deg=rng.uniform(0, 360),
                peak=_skewed_peak(rng),
            )})
    return out


# --------------------------------------------------------------------------- #
# Texture loader                                                              #
# --------------------------------------------------------------------------- #

def load_textures(folder: str | Path) -> dict:
    folder = Path(folder)
    out = {}
    if not folder.exists():
        return out
    for p in folder.iterdir():
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out[p.name] = img
    return out
