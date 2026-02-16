import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

img_size = 224

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "best_model.h5")
    
    if not os.path.exists(model_path):
        st.error("File Model 'best_model.h5' Tidak Ditemukan!")
        st.stop()
        
    return tf.keras.models.load_model(model_path)

model = load_model()

st.title("Smart Waste Classifier")
st.markdown("---")
st.subheader("Informasi Model")

st.markdown("""
**Model yang digunakan:** MobileNetV2 (Transfer Learning)  
**Ukuran Input:** 224x224 piksel  
**Output:** Binary Classification (Organik vs Daur Ulang)  
**Aktivasi Output:** Sigmoid  

Model dilatih menggunakan teknik *transfer learning* untuk meningkatkan akurasi
dan mempercepat proses training dengan memanfaatkan fitur visual dari model pretrained.
""")

st.markdown("---")

st.write("Unggah gambar untuk mengklasifikasikan sampah menjadi **Organik** atau **Daur Ulang**.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((img_size, img_size))

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Recyclable / Daur Ulang"
        confidence = float(prediction)
        guidance = "Buang sampah ini ke tempat sampah daur ulang."
    else:
        label = "Organic / Organik"
        confidence = float(1 - prediction)
        guidance = "Buang sampah ini ke tempat sampah kompos/organik."

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
    st.info(guidance)
