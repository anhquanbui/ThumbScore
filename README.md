# 🎨 YouTube Thumbnail Aesthetic Scorer (Colab Notebook)

This project shows how to **analyze and score YouTube thumbnails** by visual aesthetics using a **single Google Colab notebook** and a lightweight **Streamlit** app for interactive viewing.

- **Dataset used:** [YouTube Thumbnail Dataset by Adina Punyobanerjee (Kaggle)](https://www.kaggle.com/datasets/adinapunyobanerjee/youtube-thumbnail-dataset)  
- **Goal:** Compute an **aesthetic proxy score (0–1)** for each thumbnail (no CTR labels required).  
- **Why proxy?** The Kaggle dataset contains **images only** (no CTR/engagement). We estimate how visually appealing an image is using classical CV features and, optionally, upgrade later to a learned model (e.g., ViT).

---

## 📦 Dataset Summary

- Path layout in Kaggle/Colab environments typically includes subfolders like:  
  `.../youtube-thumbnail-dataset/youtube_thumbs/yt_thumbs/`
- Files are **JPG thumbnails** from popular YouTube channels.  
- There is **no accompanying CSV** with video metadata; the dataset is **image‑only**.

> Use cases: aesthetic quality assessment, clustering, and as input to downstream clickability models (with your own CTR/labels).

---

## 🧠 Method (Aesthetic Proxy)

For each image, we compute four robust visual features:

1. **Sharpness** — variance of Laplacian (focus measure)  
2. **Contrast** — standard deviation of grayscale intensities  
3. **Saturation** — mean **S** channel in HSV space  
4. **Edge Density** — ratio of Canny edges to image area  

We **normalize** each feature to [0, 1] using the 1st–99th percentile range (to reduce outliers), then combine them:

\[
\text{Score} = 0.35\,S_{\text{sharp}} + 0.30\,S_{\text{contrast}} + 0.25\,S_{\text{saturation}} + 0.10\,S_{\text{edges}}
\]

This produces an **aesthetic proxy score (0–1)** that you can sort/filter. It’s fast, label‑free, and a solid baseline for exploration.

> 🔮 **Upgrade path:** fine‑tune a ViT (e.g., `vit_base_patch16_384` from `timm`) on AVA / LAION‑Aesthetic labels and replace the proxy with a learned aesthetic regressor.

---

## 🚀 How to Run in Google Colab (Single Notebook)

### 1) Install dependencies
```python
!pip -q install kagglehub streamlit pyngrok==7.1.5 opencv-python pandas numpy pillow
```

### 2) Download the dataset (KaggleHub)
```python
import kagglehub, glob
path = kagglehub.dataset_download("adinapunyobanerjee/youtube-thumbnail-dataset")
print("Downloaded to:", path)

# Optional: verify a few paths
for p in glob.glob(path + "/**/*.jpg", recursive=True)[:5]:
    print(p)
```

### 3) (Optional) Compute scores directly in the notebook
```python
import cv2, numpy as np, pandas as pd, os, glob
from tqdm import tqdm

def img_stats(fp):
    img = cv2.imread(fp); 
    if img is None: return None
    h,w = img.shape[:2]; area = float(h*w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[...,1].mean()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.sum()/255.0/max(area,1.0)
    return sharp, contrast, saturation, edge_density

images = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)
rows = []
for fp in tqdm(images[:5000]):  # adjust limit if needed
    s = img_stats(fp)
    if s: rows.append((fp,)+s)

df = pd.DataFrame(rows, columns=["filepath","sharpness","contrast","saturation","edge_density"])

# Robust 1–99% normalization and weighted score
for c in ["sharpness","contrast","saturation","edge_density"]:
    q1, q99 = df[c].quantile(0.01), df[c].quantile(0.99)
    df[c+"_norm"] = (df[c].clip(q1, q99) - q1) / (q99 - q1 + 1e-8)

df["aesthetic_proxy"] = (
    0.35*df["sharpness_norm"] +
    0.30*df["contrast_norm"] +
    0.25*df["saturation_norm"] +
    0.10*df["edge_density_norm"]
)

df.sort_values("aesthetic_proxy", ascending=False).head(10)
```

### 4) Launch an interactive **Streamlit** viewer from Colab

We’ll run Streamlit **as a background server** and open it via an **ngrok** tunnel.

> 🔑 **ngrok now requires an account and personal auth token** (free tier works).
>
> 1. Sign up: https://dashboard.ngrok.com/signup  
> 2. Copy your token: https://dashboard.ngrok.com/get-started/your-authtoken  
> 3. Set the token in Colab before creating the tunnel.

```python
# (A) Save the app code to /content/streamlit_app.py (if not already saved)
# I placed this .py file in the reposity
app_code = r'''
import os, cv2, numpy as np, pandas as pd
import streamlit as st
from PIL import Image
import glob

st.set_page_config(page_title="Thumbnail Aesthetic Scorer (Proxy)", layout="wide")
st.title("🎨 Thumbnail Aesthetic Scorer — Proxy Version")

IMG_EXTS = (".jpg",".jpeg",".png",".webp",".bmp",".tiff")

def compute_df(paths):
    rows = []
    for fp in paths:
        img = cv2.imread(fp); 
        if img is None: 
            continue
        h,w = img.shape[:2]; area = float(h*w)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[...,1].mean()
        edges = cv2.Canny(gray, 100, 200)
        edge_density = edges.sum()/255.0/max(area,1.0)
        rows.append((fp,sharp,contrast,saturation,edge_density))
    df = pd.DataFrame(rows, columns=["filepath","sharpness","contrast","saturation","edge_density"])
    if df.empty: return df
    for c in ["sharpness","contrast","saturation","edge_density"]:
        q1,q99 = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c+"_norm"] = (df[c].clip(q1,q99) - q1) / (q99 - q1 + 1e-8)
    df["aesthetic_proxy"] = (
        0.35*df["sharpness_norm"] + 0.30*df["contrast_norm"]
        + 0.25*df["saturation_norm"] + 0.10*df["edge_density_norm"]
    )
    return df

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Score a folder", "Upload images"])
    limit = st.number_input("Max images", 50, 50000, 2000, step=50)
    topk = st.slider("Top-K gallery", 5, 100, 24, step=1)
    width = st.number_input("Preview width (px)", 100, 1280, 360, step=10)

if mode == "Score a folder":
    folder = st.text_input("Folder path", "/kaggle/input/youtube-thumbnail-dataset/youtube_thumbs/yt_thumbs")
    if st.button("Run"):
        if not os.path.isdir(folder):
            st.error("Folder not found.")
        else:
            imgs = []
            for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
                if p.lower().endswith(IMG_EXTS):
                    imgs.append(p)
                    if len(imgs) >= limit: break
            st.write(f"Found {len(imgs)} images.")
            df = compute_df(imgs)
            if df.empty:
                st.warning("No valid images processed.")
            else:
                st.dataframe(df.sort_values("aesthetic_proxy", ascending=False))
                st.download_button("Download CSV", df.to_csv(index=False).encode(), "thumbnail_proxy_scores.csv")
                st.subheader("Top thumbnails")
                cols = st.columns(6)
                for i, (_, r) in enumerate(df.sort_values("aesthetic_proxy", ascending=False).head(topk).iterrows()):
                    with cols[i%6]:
                        st.image(r["filepath"], width=width, caption=f"{r['aesthetic_proxy']:.3f}")
else:
    uploads = st.file_uploader("Upload images", accept_multiple_files=True,
                               type=[e.replace(".","") for e in IMG_EXTS])
    if uploads:
        tmp = "uploaded_images"; os.makedirs(tmp, exist_ok=True)
        paths = []
        for f in uploads:
            out = os.path.join(tmp, f.name)
            with open(out, "wb") as g: g.write(f.read())
            paths.append(out)
        df = compute_df(paths)
        st.dataframe(df.sort_values("aesthetic_proxy", ascending=False))
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "uploaded_thumbnail_proxy_scores.csv")
        st.subheader("Top thumbnails")
        cols = st.columns(6)
        for i, (_, r) in enumerate(df.sort_values("aesthetic_proxy", ascending=False).head(24).iterrows()):
            with cols[i%6]:
                st.image(r["filepath"], width=360, caption=f"{r['aesthetic_proxy']:.3f}")
'''
open("/content/streamlit_app.py","w").write(app_code)
```

```python
# (B) Start Streamlit (background) and open via ngrok
!pkill -f "streamlit run" || true
!fuser -k 8501/tcp || true
!nohup streamlit run /content/streamlit_app.py --server.port 8501 --server.headless true > /content/streamlit.log 2>&1 &

from pyngrok import ngrok
ngrok.set_auth_token("YOUR_AUTHTOKEN_HERE")  # <-- REQUIRED: paste your token here
public_url = ngrok.connect(8501, "http")
print("✅ Public URL:", public_url.public_url)

# Optional: tail logs for debugging
print("\n--- LOG TAIL ---")
print("\n".join(open("/content/streamlit.log").read().splitlines()[-20:]))
```

Click the **Public URL** to open your Streamlit app.

---

## 📝 Notes & Tips

- **ngrok auth token is mandatory** (even on free tier). Without it, the tunnel will fail to start.  
- If the app is unresponsive with very large folders, reduce “Max images” in the sidebar.  
- Save the scored DataFrame to CSV for later analysis or comparisons.  
- When you later have labels (e.g., AVA/LAION aesthetic scores or your own CTR), replace the proxy with a trained regressor (ViT) and keep the Streamlit UI unchanged.

---

## ⚖️ License & Usage

- This notebook/app is for **research & education** only.  
- Respect the **Kaggle dataset license** and YouTube’s Terms of Service for thumbnail usage.  
- ngrok free tunnels auto‑expire; rerun the tunnel cell to get a new URL.

**Enjoy exploring what makes a thumbnail _pop_!**

**Author:** *Anh Quan Bui*  
**Platform:** *Google Colab + Kaggle + Streamlit / `ngrok` (**requires signup & personal auth token**)  
**Date:** *October 2025*
