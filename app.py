import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import requests

# Konfigurasi halaman
st.set_page_config(page_title="Polyp Segmentation App", layout="wide")

# Path lokal untuk menyimpan model
MODEL_URL = "https://huggingface.co/Bunga1208/PengolahanSinyal_Kelompok11/resolve/main/unet_polyp_model.keras"
MODEL_PATH = "unet_polyp_model.keras"

# Fungsi unduh model sekali saja
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 Mengunduh model dari Hugging Face..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
            st.success("✅ Model berhasil diunduh!")

# Load model dengan cache
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Judul dan identitas kelompok
st.title("🩺 Polyp Segmentation using U-Net")
st.markdown("""
<div style='font-size:22px; font-weight:bold; margin-top:10px;'>
    📡 Pengolahan Sinyal Medis
</div>
<div style='font-size:20px; font-weight:bold; margin-bottom:20px;'>
    👥 Kelompok 11 - Libby, Rakha, Bunga
</div>
""", unsafe_allow_html=True)

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload an endoscopy image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        # Preprocessing
        img_resized = image.resize((256, 256))
        img_array = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_input)[0]
        mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255

        # Overlay hasil segmentasi
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(img_resized), 0.7, mask_colored, 0.3, 0)

        st.subheader("🔍 Segmentation Result")
        st.image(overlay, caption="✨ Overlay Prediction", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")
else:
    st.info("📂 Silakan upload gambar terlebih dahulu.")
