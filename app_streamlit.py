import streamlit as st
from predict import preprocess_image
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

MODEL_PATH = "../saved_models/best_model.h5"
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    info = tfds.builder('tf_flowers').info
    return info.features['label'].names

st.title("Flower Classifier — Demo")
st.markdown("Upload an image of a flower and the model will predict the species.")

uploaded = st.file_uploader("Choose an image...", type=['jpg','jpeg','png'])
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    model = load_model_cached()
    labels = load_labels()
    x = np.array(img.resize((224,224))) / 255.0
    preds = model.predict(np.expand_dims(x, axis=0))[0]
    topIdx = preds.argmax()
    st.write(f"**Prediction:** {labels[topIdx]}  —  confidence {preds[topIdx]:.3f}")
    st.write("Top 3:")
    top3 = np.argsort(preds)[-3:][::-1]
    for i in top3:
        st.write(f"- {labels[i]}: {preds[i]:.3f}")
