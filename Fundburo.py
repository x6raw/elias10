import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Titel
st.title("🔍 KI-Fundbüro")
st.write("Lade ein Bild eines gefundenen Gegenstands hoch.")

# Debug (optional)
st.write("Streamlit Version:", st.__version__)

# Modellpfad sicher laden
MODEL_PATH = os.path.join(os.path.dirname(__file__), "keras_Model.h5")

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
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen (FIX ohne Fehler)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", width=400)

    # Vorbereitung
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    # Ergebnis
    st.subheader("📊 Ergebnis")
    st.write(f"**Erkanntes Objekt:** {class_name}")
    st.progress(confidence_score)

    st.write(f"**Sicherheit:** {round(confidence_score * 100, 2)} %")

    # Fundbüro-Logik
    st.subheader("📦 Fundbüro-Eintrag")

    if confidence_score > 0.7:
        st.success(f"Gespeichert als: {class_name}")
    else:
        st.warning("Unsichere Erkennung – bitte prüfen")

    # Session Speicher
    if "items" not in st.session_state:
        st.session_state.items = []

    if st.button("➕ Ins Fundbüro aufnehmen"):
        st.session_state.items.append({
            "name": class_name,
            "confidence": confidence_score
        })
        st.success("Eintrag gespeichert!")

# Liste anzeigen
st.subheader("📋 Aktuelles Fundbüro")

if "items" in st.session_state and len(st.session_state.items) > 0:
    for i, item in enumerate(st.session_state.items):
        st.write(f"{i+1}. {item['name']} ({round(item['confidence']*100, 2)}%)")
else:
    st.info("Noch keine Einträge vorhanden.")
