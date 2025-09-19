import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from data_prep import load_flowers_tfds, IMAGE_SIZE, BATCH_SIZE

MODEL_DIR = "../saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(num_classes, input_shape=(224,224,3), base_trainable=False):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    train_ds, val_ds, test_ds, info = load_flowers_tfds()
    num_classes = info.features['label'].num_classes

    model = build_model(num_classes)
    model.summary()

    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.h5'),
                                 monitor='val_accuracy',
                                 save_best_only=True, verbose=1)

    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

    history = model.fit(train_ds,
                        epochs=15,
                        validation_data=val_ds,
                        callbacks=[checkpoint, early, reduce_lr])

    print("Evaluating on test set...")
    model.evaluate(test_ds)

if __name__ == "__main__":
    main()
