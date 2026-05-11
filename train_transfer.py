
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from visualize import save_confusion_matrix, save_class_metrics, save_class_distribution

BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "data"
TRAIN_DIR      = DATA_DIR / "train"
VAL_DIR        = DATA_DIR / "val"
TEST_DIR       = DATA_DIR / "test"
TL_MODEL_PATH  = BASE_DIR / "weather_model_tl.h5"
TL_HISTORY_PNG = BASE_DIR / "training_history_tl.png"


IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 5
SEED        = 42

CLASSES = ["cloudy", "foggy", "rainy", "shine", "sunrise"]

PHASE1_EPOCHS = 15
PHASE1_LR     = 1e-3


PHASE2_EPOCHS   = 15
PHASE2_LR       = 1e-5     
UNFREEZE_LAST_N = 30      

_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.10),
        layers.RandomZoom((-0.20, 0.20)),
        layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
        layers.RandomBrightness(factor=0.15, value_range=(0, 1)),
        layers.RandomContrast(factor=0.15),
    ],
    name="augmentation",
)

def _to_float(images, labels):
    return tf.cast(images, tf.float32), labels

def _augment(images, labels):

    imgs_01 = images / 255.0
    imgs_01 = _augmentation(imgs_01, training=True)
    return imgs_01 * 255.0, labels


def _mobilenet_preprocess(images, labels):

    return mobilenet_preprocess(images), labels


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
    ds = ds.map(_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_mobilenet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_weights() -> dict:
    counts  = np.array(
        [len(list((TRAIN_DIR / cls).iterdir())) for cls in CLASSES], dtype=int
    )
    y_flat  = np.repeat(np.arange(NUM_CLASSES), counts)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_flat,
    )
    weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print("\nClass weights:")
    for i, cls in enumerate(CLASSES):
        print(f"  [{i}] {cls:<10}  weight={weight_dict[i]:.4f}  ({counts[i]} samples)")
    return weight_dict

def build_transfer_model():
  

    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input")

    x = base_model(inputs, training=False)

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.50, name="dropout")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="WeatherCNN_TL")
    return model, base_model


def unfreeze_top_n_layers(base_model: keras.Model, n: int) -> None:
  
    base_model.trainable = True  

    for layer in base_model.layers[:-n]:
        layer.trainable = False

    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n  Backbone: {n_trainable}/{len(base_model.layers)} layers unfrozen")

def save_history_plot(h1, h2) -> None:

    acc      = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss     = h1.history["loss"]         + h2.history["loss"]
    val_loss = h1.history["val_loss"]     + h2.history["val_loss"]
    p1_end   = len(h1.history["accuracy"])  # x-position of phase boundary

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(acc,    label="Train Accuracy")
    axes[0].plot(val_acc, label="Val Accuracy")
    axes[0].axvline(p1_end - 0.5, color="red", linestyle="--", label="Phase 2 start")
    axes[0].set_title("Transfer Learning — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(loss,     label="Train Loss")
    axes[1].plot(val_loss, label="Val Loss")
    axes[1].axvline(p1_end - 0.5, color="red", linestyle="--", label="Phase 2 start")
    axes[1].set_title("Transfer Learning — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(str(TL_HISTORY_PNG), dpi=150)
    plt.close(fig)
    print(f"History plot saved -> {TL_HISTORY_PNG}")


def print_comparison(tl_metrics: dict, phase1_params: int, phase2_params: int) -> None:

    scratch_path = BASE_DIR / "scratch_metrics.json"
    s = {}
    if scratch_path.exists():
        with open(scratch_path) as f:
            s = json.load(f)
    else:
        print("\n[INFO] scratch_metrics.json not found.")
        print("       Run train.py first to populate scratch CNN metrics.\n")

    def fmt_pct(val):
        return f"{val * 100:.2f}%" if val is not None else "N/A"

    def fmt_time(val):
        return f"{val:.0f}s" if val is not None else "N/A"

    def fmt_int(val):
        return f"{val:,}" if val is not None else "N/A"

    rows = [
        ("Metric",                     "Scratch CNN",                           "Transfer (MobileNetV2)"),
        ("-" * 35,                     "-" * 20,                                "-" * 22),
        ("Best Training Accuracy",     fmt_pct(s.get("best_train_accuracy")),   fmt_pct(tl_metrics["best_train_accuracy"])),
        ("Best Validation Accuracy",   fmt_pct(s.get("best_val_accuracy")),     fmt_pct(tl_metrics["best_val_accuracy"])),
        ("Final Test Accuracy",        fmt_pct(s.get("test_accuracy")),         fmt_pct(tl_metrics["test_accuracy"])),
        ("Total Training Time",        fmt_time(s.get("training_time_s")),      fmt_time(tl_metrics["training_time_s"])),
        ("Trainable Params (Phase 1)", fmt_int(s.get("trainable_params")),      fmt_int(phase1_params)),
        ("Trainable Params (Phase 2)", "N/A",                                   fmt_int(phase2_params)),
    ]

    print("\n" + "=" * 82)
    print("  MODEL COMPARISON: Scratch CNN vs. Transfer Learning (MobileNetV2)")
    print("=" * 82)
    for label, s_val, t_val in rows:
        print(f"  {label:<35} {s_val:<22} {t_val}")
    print("=" * 82)

if __name__ == "__main__":

    for d, name in [(TRAIN_DIR, "train"), (VAL_DIR, "val"), (TEST_DIR, "test")]:
        if not d.exists():
            raise FileNotFoundError(
                f"[ERROR] '{name}' directory not found: {d}\n"
                "Run prepare_dataset.py first."
            )

    print("Loading datasets (224x224 for MobileNetV2) ...")
    train_ds = load_dataset(TRAIN_DIR, shuffle=True,  augment=True)
    val_ds   = load_dataset(VAL_DIR,   shuffle=False, augment=False)
    test_ds  = load_dataset(TEST_DIR,  shuffle=False, augment=False)

    class_weights = compute_weights()

    print("\nBuilding transfer learning model ...")
    model, base_model = build_transfer_model()
    model.summary()

    print("\n" + "=" * 65)
    print("  PHASE 1: Training custom head (backbone fully frozen)")
    print("=" * 65)

    phase1_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    print(f"  Trainable parameters: {phase1_params:,}")
    print(f"  Learning rate: {PHASE1_LR}")
    print(f"  Max epochs: {PHASE1_EPOCHS}\n")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(TL_MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    t_start = time.time()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=phase1_callbacks,
        class_weight=class_weights,
    )


    print("\n" + "=" * 65)
    print(f"  PHASE 2: Fine-tuning last {UNFREEZE_LAST_N} backbone layers")
    print("=" * 65)

    unfreeze_top_n_layers(base_model, UNFREEZE_LAST_N)

    phase2_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    print(f"  Trainable parameters: {phase2_params:,}")
    print(f"  Learning rate: {PHASE2_LR}  (low LR protects pretrained weights)")
    print(f"  Max epochs: {PHASE2_EPOCHS}\n")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase2_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(TL_MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=phase2_callbacks,
        class_weight=class_weights,
    )

    training_time = time.time() - t_start

    save_history_plot(h1, h2)

    print("\n" + "=" * 65)
    print("  FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 65)
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")

    all_train_acc = h1.history["accuracy"]     + h2.history["accuracy"]
    all_val_acc   = h1.history["val_accuracy"] + h2.history["val_accuracy"]

    tl_metrics = {
        "best_train_accuracy": float(max(all_train_acc)),
        "best_val_accuracy":   float(max(all_val_acc)),
        "test_accuracy":       float(test_acc),
        "test_loss":           float(test_loss),
        "training_time_s":     float(training_time),
        "phase1_params":       phase1_params,
        "phase2_params":       phase2_params,
    }
    with open(BASE_DIR / "tl_metrics.json", "w") as f:
        json.dump(tl_metrics, f, indent=2)

    print_comparison(tl_metrics, phase1_params, phase2_params)
    print(f"\nBest TL model saved -> {TL_MODEL_PATH}")

    save_class_distribution(TRAIN_DIR, VAL_DIR, TEST_DIR,
                            BASE_DIR / "class_distribution.png")
    y_true, y_pred = save_confusion_matrix(
        model, test_ds,
        BASE_DIR / "confusion_matrix_tl.png",
        title="Confusion Matrix - Transfer Learning / MobileNetV2 (Test Set)",
    )
    save_class_metrics(
        y_true, y_pred,
        BASE_DIR / "class_metrics_tl.png",
        title="Per-class Metrics - Transfer Learning / MobileNetV2 (Test Set)",
    )
