

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import time
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import keras
from keras import layers
from visualize import save_confusion_matrix, save_class_metrics, save_class_distribution



BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
TEST_DIR    = DATA_DIR / "test"
MODEL_PATH  = BASE_DIR / "weather_model.h5"
HISTORY_PNG = BASE_DIR / "training_history.png"

IMG_SIZE    = (150, 150)
BATCH_SIZE  = 32
EPOCHS      = 50      
NUM_CLASSES = 5
SEED        = 42

CLASSES = ["cloudy", "foggy", "rainy", "shine", "sunrise"]



_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.10),            # +-36 degrees
        layers.RandomZoom((-0.20, 0.20)),        # zoom in or out up to 20%
        layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
        layers.RandomBrightness(factor=0.15, value_range=(0, 1)),
        layers.RandomContrast(factor=0.15),
    ],
    name="augmentation",
)


def _normalize(images, labels):

    return tf.cast(images, tf.float32) / 255.0, labels


def _augment(images, labels):

    return _augmentation(images, training=True), labels


def load_dataset(directory: Path, *, shuffle: bool, augment: bool) -> tf.data.Dataset:

    ds = keras.utils.image_dataset_from_directory(
        str(directory),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=CLASSES,
        shuffle=shuffle,
        seed=SEED,
    )
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)



def compute_weights() -> dict:

    counts = np.array(
        [len(list((TRAIN_DIR / cls).iterdir())) for cls in CLASSES],
        dtype=int,
    )
    y_flat  = np.repeat(np.arange(NUM_CLASSES), counts)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_flat,
    )
    weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print("\nClass weights (imbalance correction):")
    for i, cls in enumerate(CLASSES):
        print(f"  [{i}] {cls:<10}  weight={weight_dict[i]:.4f}  ({counts[i]} samples)")
    return weight_dict



def build_model() -> keras.Model:

    reg = keras.regularizers.l2(1e-4)

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input")


    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

 
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg, name="fc1")(x)
    x = layers.Dropout(0.40, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="WeatherCNN_v2")
    return model



def save_history_plot(history: keras.callbacks.History) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(str(HISTORY_PNG), dpi=150)
    plt.close(fig)
    print(f"Training history plot saved -> {HISTORY_PNG}")


if __name__ == "__main__":


    for d, name in [(TRAIN_DIR, "train"), (VAL_DIR, "val"), (TEST_DIR, "test")]:
        if not d.exists():
            raise FileNotFoundError(
                f"[ERROR] '{name}' directory not found: {d}\n"
                "Run prepare_dataset.py first."
            )


    print("Loading datasets ...")
    train_ds = load_dataset(TRAIN_DIR, shuffle=True,  augment=True)
    val_ds   = load_dataset(VAL_DIR,   shuffle=False, augment=False)
    test_ds  = load_dataset(TEST_DIR,  shuffle=False, augment=False)

    class_weights = compute_weights()


    print("\nBuilding model ...")
    model = build_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),

        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


    print(f"\nStarting training (max {EPOCHS} epochs, early stopping enabled) ...\n")
    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    training_time = time.time() - t_start

    save_history_plot(history)

    
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 50)
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print("=" * 50)
    print(f"\nBest model saved -> {MODEL_PATH}")

    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    scratch_metrics = {
        "best_train_accuracy": float(max(history.history["accuracy"])),
        "best_val_accuracy":   float(max(history.history["val_accuracy"])),
        "test_accuracy":       float(test_acc),
        "test_loss":           float(test_loss),
        "training_time_s":     float(training_time),
        "trainable_params":    trainable_params,
    }
    with open(BASE_DIR / "scratch_metrics.json", "w") as f:
        json.dump(scratch_metrics, f, indent=2)
    print(f"Metrics saved -> {BASE_DIR / 'scratch_metrics.json'}")

    save_class_distribution(TRAIN_DIR, VAL_DIR, TEST_DIR,
                            BASE_DIR / "class_distribution.png")
    y_true, y_pred = save_confusion_matrix(
        model, test_ds,
        BASE_DIR / "confusion_matrix.png",
        title="Confusion Matrix - Scratch CNN (Test Set)",
    )
    save_class_metrics(
        y_true, y_pred,
        BASE_DIR / "class_metrics.png",
        title="Per-class Metrics - Scratch CNN (Test Set)",
    )
