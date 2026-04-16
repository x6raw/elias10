import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

st.title("🔍 KI-Fundbüro")

# Modell laden
MODEL_PATH = "keras_Model.h5"

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

# Labels laden
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return f.readlines()

class_names = load_labels()

# Upload
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # ❗ KEIN use_container_width mehr
    st.image(image, width=300)

    # Vorbereitung
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized

    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index].strip()
    confidence = prediction[0][index]

    st.write("Ergebnis:", class_name)
    st.write("Sicherheit:", round(confidence * 100, 2), "%")
