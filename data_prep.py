import tensorflow as tf
import tensorflow_datasets as tfds
import os

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def load_flowers_tfds(split_ratio=(0.8, 0.1, 0.1), batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    dataset_name = "tf_flowers"
    ds, ds_info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    all_ds = ds['train']
    total = ds_info.splits['train'].num_examples

    train_count = int(total * split_ratio[0])
    val_count = int(total * split_ratio[1])
    test_count = total - train_count - val_count

    all_ds = all_ds.shuffle(1024, seed=123)

    train_ds = all_ds.take(train_count)
    rest = all_ds.skip(train_count)
    val_ds = rest.take(val_count)
    test_ds = rest.skip(val_count)

    def preprocess(image, label):
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, ds_info

if __name__ == "__main__":
    train_ds, val_ds, test_ds, info = load_flowers_tfds()
    print("Dataset loaded. Classes:", info.features['label'].num_classes)
