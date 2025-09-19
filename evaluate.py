import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_prep import load_flowers_tfds
from tensorflow.keras.models import load_model

MODEL_PATH = "../saved_models/best_model.h5"

def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    train_ds, val_ds, test_ds, info = load_flowers_tfds()
    class_names = info.features['label'].names

    model = load_model(MODEL_PATH)

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized Confusion Matrix')

if __name__ == "__main__":
    main()
