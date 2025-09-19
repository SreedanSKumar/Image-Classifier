import tensorflow as tf
import numpy as np
from PIL import Image
import sys
from tensorflow.keras.models import load_model
from data_prep import IMAGE_SIZE
import tensorflow_datasets as tfds

MODEL_PATH = "../saved_models/best_model.h5"

def load_labels():
    info = tfds.builder('tf_flowers').info
    return info.features['label'].names

def preprocess_image(img_path, image_size=IMAGE_SIZE):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(image_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img_path):
    model = load_model(MODEL_PATH)
    labels = load_labels()
    x = preprocess_image(img_path)
    preds = model.predict(x)[0]
    top_idx = preds.argmax()
    print("Prediction:", labels[top_idx], f"(confidence: {preds[top_idx]:.3f})")
    top3 = np.argsort(preds)[-3:][::-1]
    for i in top3:
        print(f"{labels[i]}: {preds[i]:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py /path/to/image.jpg")
    else:
        predict(sys.argv[1])
