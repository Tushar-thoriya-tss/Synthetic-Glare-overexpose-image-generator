"""Streamlit UI for synthetic glare/overexposure generation.

Run:
    streamlit run app.py

Modes:
  - Auto-randomize + review : tool generates random glares per image; you tweak.
  - Manual placement        : draw shapes on the image (click/drag) to place glare.

Outputs (per saved image):
  <output>/images/<stem>_glare.png
  <output>/masks/<stem>_mask.png   (binary 0/255, single channel)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from glare_engine import compose, load_textures, random_glares

# --- Compat shim: streamlit-drawable-canvas (<=0.9.3) calls
# `streamlit.elements.image.image_to_url(image, width:int, clamp, channels,
# output_format, image_id)`, but Streamlit ≥1.36 moved/redesigned that helper
# (now expects a `layout_config` object). We bypass Streamlit entirely and
# return a base64 data URL the browser can render directly.
import base64
import io as _io
import streamlit.elements.image as _st_image_mod

def _image_to_url_compat(image, width=None, clamp=False, channels="RGB",
                         output_format="PNG", image_id=""):
    pil = image
    if not isinstance(pil, Image.Image):
        pil = Image.fromarray(pil)
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGB")
    fmt = (output_format or "PNG").upper()
    buf = _io.BytesIO()
    pil.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

_st_image_mod.image_to_url = _image_to_url_compat

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False


# --------------------------------------------------------------------------- #
# Session-state init                                                          #
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "input_folder": "input_images",
    "output_folder": "output",
    "textures_folder": "glare_textures",
    "image_files": [],
    "img_idx": 0,
    "current_path": None,
    "current_image": None,   # numpy RGB
    "glares": [],            # list of glare dicts
    "canvas_key": 0,         # bump to clear canvas
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.set_page_config(page_title="Glare Generator", layout="wide")


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def list_images(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS])


def load_current_image():
    files = st.session_state.image_files
    if not files:
        st.session_state.current_image = None
        st.session_state.current_path = None
        return
    idx = st.session_state.img_idx % len(files)
    p = files[idx]
    if st.session_state.current_path != p:
        bgr = cv2.imread(str(p))
        if bgr is None:
            return
        st.session_state.current_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        st.session_state.current_path = p
        st.session_state.glares = []
        st.session_state.canvas_key += 1


def save_outputs(image_rgb, mask, src_path, out_folder, jpeg_quality=95):
    """Save glared image (in source format) and binary mask (always PNG).

    Both outputs are guaranteed to match the source image's H×W exactly.
    """
    src_path = Path(src_path)
    src_h, src_w = image_rgb.shape[:2]

    # Dimension safety net — should never fire, but catches future regressions.
    assert image_rgb.shape[:2] == mask.shape[:2] == (src_h, src_w), (
        f"shape mismatch — img {image_rgb.shape[:2]} mask {mask.shape[:2]} "
        f"src {(src_h, src_w)}"
    )

    out_images = Path(out_folder) / "images"
    out_masks = Path(out_folder) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    stem = src_path.stem
    ext = src_path.suffix.lower() or ".png"
    img_out = out_images / f"{stem}_glare{ext}"
    mask_out = out_masks / f"{stem}_mask.png"

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(img_out), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    elif ext == ".webp":
        cv2.imwrite(str(img_out), bgr, [cv2.IMWRITE_WEBP_QUALITY, int(jpeg_quality)])
    else:
        # PNG / BMP / TIFF — lossless, no quality flag needed
        cv2.imwrite(str(img_out), bgr)

    cv2.imwrite(str(mask_out), mask)
    return img_out, mask_out


def step_image(delta: int):
    n = max(1, len(st.session_state.image_files))
    st.session_state.img_idx = (st.session_state.img_idx + delta) % n
    st.session_state.current_path = None
    load_current_image()


# --------------------------------------------------------------------------- #
# Canvas → glare conversion (manual mode)                                     #
# --------------------------------------------------------------------------- #
def canvas_obj_to_glare(obj: dict, tool: str, inv_scale: float, params: dict):
    """Convert a fabric.js object back into an image-space glare dict."""
    typ = obj.get("type")
    if tool == "ellipse" and typ == "circle":
        radius = obj["radius"]
        sx = obj.get("scaleX", 1.0); sy = obj.get("scaleY", 1.0)
        cx = (obj["left"] + radius * sx) * inv_scale
        cy = (obj["top"] + radius * sy) * inv_scale
        rx = max(radius * sx * inv_scale, 2.0)
        ry = max(radius * sy * inv_scale, 2.0)
        return {"type": "ellipse", "params": dict(
            cx=cx, cy=cy, rx=rx, ry=ry,
            angle_deg=obj.get("angle", 0.0),
            peak=params["peak"], softness=params["softness"],
        )}

    if tool == "line" and typ == "line":
        left = obj.get("left", 0); top = obj.get("top", 0)
        x1, y1 = obj.get("x1", 0), obj.get("y1", 0)
        x2, y2 = obj.get("x2", 0), obj.get("y2", 0)
        return {"type": "line", "params": dict(
            x1=(left + x1) * inv_scale, y1=(top + y1) * inv_scale,
            x2=(left + x2) * inv_scale, y2=(top + y2) * inv_scale,
            width=params["line_width"] * inv_scale,
            peak=params["peak"], softness=params["softness"],
        )}

    if tool in ("polygon", "freedraw") and typ == "path":
        path = obj.get("path", [])
        pts = []
        for cmd in path:
            if not cmd:
                continue
            op = cmd[0]
            if op in ("M", "L") and len(cmd) >= 3:
                pts.append([cmd[1] * inv_scale, cmd[2] * inv_scale])
            elif op == "Q" and len(cmd) >= 5:
                pts.append([cmd[3] * inv_scale, cmd[4] * inv_scale])
        if len(pts) < 2:
            return None
        if tool == "polygon":
            return {"type": "polygon", "params": dict(
                points=pts, peak=params["peak"],
                softness=params["softness"],
                blur_sigma=params["blur_sigma"],
            )}
        return {"type": "freedraw", "params": dict(
            points=pts,
            brush=params["brush"] * inv_scale,
            peak=params["peak"], softness=params["softness"],
        )}

    if tool == "texture" and typ == "circle":
        radius = obj.get("radius", 3)
        cx = (obj["left"] + radius) * inv_scale
        cy = (obj["top"] + radius) * inv_scale
        return {"type": "texture", "params": dict(
            texture_name=params["texture_name"],
            cx=cx, cy=cy,
            tw=params["tex_w"], th=params["tex_h"],
            angle_deg=params["tex_angle"], peak=params["peak"],
        )}
    return None


# --------------------------------------------------------------------------- #
# Sidebar — folders & global rendering settings                               #
# --------------------------------------------------------------------------- #
st.sidebar.title("Glare Generator")

st.session_state.input_folder = st.sidebar.text_input(
    "Input folder", st.session_state.input_folder)
st.session_state.output_folder = st.sidebar.text_input(
    "Output folder", st.session_state.output_folder)
st.session_state.textures_folder = st.sidebar.text_input(
    "Textures folder (optional)", st.session_state.textures_folder)

if st.sidebar.button("Reload folder"):
    st.session_state.image_files = list_images(st.session_state.input_folder)
    st.session_state.img_idx = 0
    st.session_state.current_path = None
    load_current_image()

if not st.session_state.image_files:
    st.session_state.image_files = list_images(st.session_state.input_folder)
load_current_image()

st.sidebar.write(f"**Images found:** {len(st.session_state.image_files)}")

textures = load_textures(st.session_state.textures_folder)
texture_names = list(textures.keys())
st.sidebar.write(f"**Textures loaded:** {len(textures)}")

mode = st.sidebar.radio("Mode",
                        ["Auto-randomize + review", "Manual placement"], index=0)

st.sidebar.markdown("### Mask & blending")
mask_threshold = st.sidebar.slider(
    "Mask threshold (intensity)", 1, 200, 30,
    help="Pixels where added glare brightness ≥ threshold are marked as glare.")
blend_label = st.sidebar.selectbox(
    "Blend mode",
    ["add (clipped — true overexposure)", "screen (softer)"])
blend_key = "add" if blend_label.startswith("add") else "screen"

st.sidebar.markdown("### Color tint (RGB multipliers)")
tr = st.sidebar.slider("R", 0.5, 1.5, 1.0, 0.05)
tg = st.sidebar.slider("G", 0.5, 1.5, 1.0, 0.05)
tb = st.sidebar.slider("B", 0.5, 1.5, 1.0, 0.05)
noise_std = st.sidebar.slider("Glare-region noise (std)", 0.0, 30.0, 5.0)


# --------------------------------------------------------------------------- #
# Main area                                                                   #
# --------------------------------------------------------------------------- #
if st.session_state.current_image is None:
    st.warning(f"No images found. Put files in `{st.session_state.input_folder}/` "
               "and click **Reload folder**.")
    st.stop()

img = st.session_state.current_image
h, w = img.shape[:2]
total = len(st.session_state.image_files)
st.subheader(f"[{st.session_state.img_idx + 1}/{total}]  "
             f"{Path(st.session_state.current_path).name}   ({w}×{h})")

# Navigation row
ncol1, ncol2, ncol3, ncol4, ncol5 = st.columns(5)
with ncol1:
    if st.button("⟵ Prev"):
        step_image(-1); st.rerun()
with ncol2:
    if st.button("Skip ⟶"):
        step_image(+1); st.rerun()
with ncol3:
    if st.button("🗑 Clear glares"):
        st.session_state.glares = []
        st.session_state.canvas_key += 1
        st.rerun()
with ncol4:
    if st.button("💾 Save & next"):
        if st.session_state.glares:
            final, mask = compose(
                img, st.session_state.glares,
                mask_threshold=mask_threshold, tint=(tr, tg, tb),
                noise_std=noise_std, textures=textures, blend=blend_key)
            img_out, mask_out = save_outputs(
                final, mask, st.session_state.current_path,
                st.session_state.output_folder)
            st.success(f"Saved → {img_out.name}  +  {mask_out.name}")
            step_image(+1); st.rerun()
        else:
            st.warning("No glares placed — nothing to save.")
with ncol5:
    st.write(f"Glares on this image: **{len(st.session_state.glares)}**")


# ---------------- Mode-specific controls ---------------- #
if mode.startswith("Auto"):
    a1, a2, a3 = st.columns([1, 1, 1])
    with a1:
        n_random = st.number_input("# glares to generate", 1, 10, 2)
    with a2:
        if st.button("🎲 Generate random"):
            st.session_state.glares = random_glares(
                h, w, n=int(n_random), texture_names=texture_names)
            st.rerun()
    with a3:
        if st.button("➕ Add 1 random"):
            st.session_state.glares.extend(
                random_glares(h, w, n=1, texture_names=texture_names))
            st.rerun()

else:  # Manual
    if not HAS_CANVAS:
        st.error("`streamlit-drawable-canvas` is not installed. "
                 "Install it and reload:\n\n"
                 "    pip install streamlit-drawable-canvas")
        st.stop()

    tool_label = st.radio(
        "Tool",
        ["Ellipse (circle)", "Line / streak", "Polygon",
         "Freedraw", "Place texture (point)"],
        horizontal=True)
    tool_map = {
        "Ellipse (circle)": ("ellipse", "circle"),
        "Line / streak":    ("line",    "line"),
        "Polygon":          ("polygon", "polygon"),
        "Freedraw":         ("freedraw", "freedraw"),
        "Place texture (point)": ("texture", "point"),
    }
    tool, draw_mode = tool_map[tool_label]

    with st.expander("Parameters for next stroke", expanded=True):
        peak = st.slider("Peak brightness (added 0–300)", 30, 300, 180)
        softness = st.slider("Softness / falloff", 0.1, 1.5, 0.7, 0.05)
        line_width = st.slider("Streak width (px)", 4, 80, 16) \
            if tool == "line" else 16
        blur_sigma = st.slider("Polygon edge blur σ", 1.0, 50.0, 12.0) \
            if tool == "polygon" else 12.0
        brush = st.slider("Brush radius (freedraw)", 2, 60, 12) \
            if tool == "freedraw" else 12
        tex_name, tex_w, tex_h, tex_angle = None, int(w * 0.25), int(h * 0.25), 0
        if tool == "texture":
            if not texture_names:
                st.warning("No textures in textures folder.")
            else:
                tex_name = st.selectbox("Texture", texture_names)
                tex_w = st.slider("Texture width", 20, w, int(w * 0.25))
                tex_h = st.slider("Texture height", 20, h, int(h * 0.25))
                tex_angle = st.slider("Texture angle (deg)", 0, 360, 0)

    next_params = dict(peak=peak, softness=softness,
                       line_width=line_width, blur_sigma=blur_sigma,
                       brush=brush, texture_name=tex_name,
                       tex_w=tex_w, tex_h=tex_h, tex_angle=tex_angle)

    # Display canvas at limited width — keep scale to map back to image space.
    MAX_CANVAS_W = 900
    scale = min(1.0, MAX_CANVAS_W / w)
    disp_w, disp_h = int(w * scale), int(h * scale)
    bg = Image.fromarray(img).resize((disp_w, disp_h))

    canvas_result = st_canvas(
        fill_color="rgba(255, 220, 0, 0.35)",
        stroke_width=2,
        stroke_color="rgba(255, 220, 0, 0.95)",
        background_image=bg,
        update_streamlit=True,
        width=disp_w,
        height=disp_h,
        drawing_mode=draw_mode,
        key=f"canvas_{st.session_state.canvas_key}",
    )

    cols_apply = st.columns([1, 1, 3])
    with cols_apply[0]:
        if st.button("✅ Apply drawn shapes"):
            if canvas_result.json_data is not None:
                inv = 1.0 / scale
                added = 0
                for o in canvas_result.json_data.get("objects", []):
                    g = canvas_obj_to_glare(o, tool, inv, next_params)
                    if g is not None:
                        st.session_state.glares.append(g)
                        added += 1
                if added:
                    st.session_state.canvas_key += 1
                    st.success(f"Added {added} glare(s).")
                    st.rerun()
                else:
                    st.warning("No matching shape found on canvas.")
    with cols_apply[1]:
        if st.button("↺ Reset canvas"):
            st.session_state.canvas_key += 1
            st.rerun()


# ---------------- Live preview ---------------- #
final, mask = compose(img, st.session_state.glares,
                      mask_threshold=mask_threshold, tint=(tr, tg, tb),
                      noise_std=noise_std, textures=textures, blend=blend_key)

p1, p2, p3 = st.columns(3)
with p1:
    st.caption("Original")
    st.image(img, width="stretch")
with p2:
    st.caption("Glared (preview)")
    st.image(final, width="stretch")
with p3:
    st.caption("Binary mask")
    st.image(mask, width="stretch", clamp=True)


# ---------------- Glare list (edit/delete) ---------------- #
if st.session_state.glares:
    with st.expander(f"Placed glares ({len(st.session_state.glares)}) — edit / delete",
                     expanded=False):
        to_delete = None
        for i, g in enumerate(st.session_state.glares):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.write(f"**#{i+1}**")
                st.write(f"`{g['type']}`")
            with c2:
                # Editable peak/softness for any glare that exposes them
                p = g["params"]
                if "peak" in p:
                    p["peak"] = st.slider(
                        f"peak #{i+1}", 10, 300, int(p["peak"]),
                        key=f"peak_{i}")
                if "softness" in p:
                    p["softness"] = st.slider(
                        f"softness #{i+1}", 0.1, 1.5, float(p["softness"]), 0.05,
                        key=f"soft_{i}")
            with c3:
                if st.button("Delete", key=f"del_{i}"):
                    to_delete = i
        if to_delete is not None:
            st.session_state.glares.pop(to_delete)
            st.rerun()
