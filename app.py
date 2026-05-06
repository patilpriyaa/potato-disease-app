import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

MODEL_URL = "https://drive.google.com/uc?id=1KuV4ZfznmsM73EJz-nPiX7byGnGQviWT"
MODEL_PATH = "potato_disease_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... please wait ⏳"):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

classes = ["Early Blight", "Late Blight", "Healthy"]

st.title("🌿 Potato Leaf Disease Detection")
st.write("Upload a potato leaf image to detect disease")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image")

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    result = classes[np.argmax(pred)]

    st.success(f"Prediction: {result}")