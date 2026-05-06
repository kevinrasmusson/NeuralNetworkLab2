import tensorflow as tf

def load_data(file_path):
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    SEED = 123

    train_ds = tf.keras.utils.image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="categorical"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    print(f"Class names: {class_names}")

    return train_ds, test_ds, class_names