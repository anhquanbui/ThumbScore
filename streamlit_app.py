
# app/streamlit_app.py
# Streamlit app to score YouTube thumbnails by an aesthetic PROXY (no labels required).
# Works well inside Colab using ColabCode tunneling.
#
# Features:
# - Score a folder of images (recursive) OR upload images.
# - Compute proxy features: sharpness (Laplacian variance), contrast (std gray),
#   saturation (mean S in HSV), edge density (Canny edges / area).
# - Normalize features robustly and combine into a 0–1 proxy score.
# - Show sortable table + gallery (top-K) with scores.
#
# Note: This is a lightweight baseline. You can later replace the proxy with a ViT model score.

import os
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from typing import List, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

st.set_page_config(page_title="Thumbnail Aesthetic Scorer (Proxy)", layout="wide")

st.title("🎨 Thumbnail Aesthetic Scorer — Proxy Version")
st.caption("No labels needed • Sharpness + Contrast + Saturation + Edge density → normalized 0–1 score")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Select mode:", ["Score a folder", "Upload images"])
    img_limit = st.number_input("Max images to process", 50, 50000, 5000, step=50)
    top_k = st.slider("Top-K to show in gallery", 5, 100, 24, step=1)
    resize_w = st.number_input("Preview width (px)", 100, 1280, 360, step=10)
    st.markdown("---")
    st.markdown("**Weights** (for future model use): not required for proxy.")
    st.markdown("---")
    st.caption("Tip: For Colab, run with `ColabCode(port=8501)` to get a public URL.")

def list_images_recursive(root: str, limit: int) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, fn))
                if len(paths) >= limit:
                    return paths
    return paths

@st.cache_data(show_spinner=False)
def compute_stats_for_paths(paths: List[str]) -> pd.DataFrame:
    rows = []
    for fp in paths:
        try:
            img = cv2.imread(fp)
            if img is None:
                continue
            h, w = img.shape[:2]
            area = float(h * w)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = gray.std()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[..., 1].mean()
            edges = cv2.Canny(gray, 100, 200)
            edge_density = edges.sum() / 255.0 / max(area, 1.0)
            rows.append((fp, sharp, contrast, saturation, edge_density))
        except Exception as e:
            # Skip problematic file
            continue
    df = pd.DataFrame(rows, columns=["filepath", "sharpness", "contrast", "saturation", "edge_density"])
    if df.empty:
        return df
    # Robust normalization (1–99 percentile) per feature
    for c in ["sharpness", "contrast", "saturation", "edge_density"]:
        q1, q99 = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c + "_norm"] = (df[c].clip(q1, q99) - q1) / (q99 - q1 + 1e-8)
    # Weighted score
    df["aesthetic_proxy"] = (
        0.35 * df["sharpness_norm"] +
        0.30 * df["contrast_norm"] +
        0.25 * df["saturation_norm"] +
        0.10 * df["edge_density_norm"]
    )
    return df

def show_gallery(df: pd.DataFrame, k: int, preview_w: int):
    cols = st.columns(6)
    for i, (_, row) in enumerate(df.head(k).iterrows()):
        c = cols[i % 6]
        with c:
            st.image(row["filepath"], width=preview_w, caption=f"{row['aesthetic_proxy']:.3f}")
            st.caption(os.path.basename(row["filepath"]))

if mode == "Score a folder":
    folder = st.text_input("Folder path (absolute or relative to working dir):", value="/kaggle/input/youtube-thumbnail-dataset/youtube_thumbs/yt_thumbs")
    go = st.button("🔎 Scan & Score")
    if go:
        if not os.path.isdir(folder):
            st.error("Folder not found. Please check the path.")
        else:
            with st.status("Scanning images…", expanded=False):
                paths = list_images_recursive(folder, limit=int(img_limit))
            st.write(f"Found **{len(paths)}** images (showing up to your limit).")
            with st.status("Computing proxy scores…", expanded=False):
                df = compute_stats_for_paths(paths)
            if df.empty:
                st.warning("No valid images were processed.")
            else:
                st.success("Done.")
                st.dataframe(df.sort_values("aesthetic_proxy", ascending=False), use_container_width=True)
                st.download_button("⬇️ Download scores CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name="thumbnail_proxy_scores.csv", mime="text/csv")
                st.markdown("---")
                st.subheader("🏆 Top thumbnails by proxy score")
                show_gallery(df.sort_values("aesthetic_proxy", ascending=False), k=int(top_k), preview_w=int(resize_w))

elif mode == "Upload images":
    files = st.file_uploader("Drop one or more images", type=[e.replace(".","") for e in IMG_EXTS], accept_multiple_files=True)
    if files:
        # Save to temp folder in session dir
        tmp_root = "uploaded_images"
        os.makedirs(tmp_root, exist_ok=True)
        paths = []
        for f in files:
            # Ensure unique name
            out = os.path.join(tmp_root, f.name)
            with open(out, "wb") as g:
                g.write(f.read())
            paths.append(out)
        df = compute_stats_for_paths(paths)
        if df.empty:
            st.warning("No valid images were processed.")
        else:
            st.dataframe(df.sort_values("aesthetic_proxy", ascending=False), use_container_width=True)
            st.download_button("⬇️ Download scores CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="uploaded_thumbnail_proxy_scores.csv", mime="text/csv")
            st.markdown("---")
            st.subheader("🏆 Top thumbnails by proxy score")
            show_gallery(df.sort_values("aesthetic_proxy", ascending=False), k=int(top_k), preview_w=int(resize_w))
    else:
        st.info("Upload images to begin.")
