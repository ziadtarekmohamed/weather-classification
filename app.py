
import io
import numpy as np
from pathlib import Path
from typing import cast
from flask import Flask, request, render_template, jsonify
from PIL import Image
import keras
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


BASE_DIR         = Path(__file__).resolve().parent
SCRATCH_MODEL    = BASE_DIR / "weather_model.h5"
TL_MODEL         = BASE_DIR / "weather_model_tl.h5"

# Each model requires a different input size
SCRATCH_IMG_SIZE = (150, 150)
TL_IMG_SIZE = (150, 150)

CLASSES = ["cloudy", "foggy", "rainy", "shine", "sunrise"]


app = Flask(__name__)


_scratch_model = None
_tl_model      = None


def get_scratch_model() -> keras.Model:
    global _scratch_model
    if _scratch_model is None:
        if not SCRATCH_MODEL.exists():
            raise FileNotFoundError(
                f"Scratch model not found at {SCRATCH_MODEL}. "
                "Run train.py first."
            )
        _scratch_model = cast(keras.Model, keras.models.load_model(str(SCRATCH_MODEL)))
    return _scratch_model


def get_tl_model() -> keras.Model:
    global _tl_model
    if _tl_model is None:
        if not TL_MODEL.exists():
            raise FileNotFoundError(
                f"Transfer model not found at {TL_MODEL}. "
                "Run train_transfer.py first."
            )
        _tl_model = cast(keras.Model, keras.models.load_model(str(TL_MODEL)))
    return _tl_model

def preprocess_for_scratch(pil_img: Image.Image) -> np.ndarray:

    img = pil_img.resize(SCRATCH_IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return cast(np.ndarray, np.expand_dims(arr, axis=0))        # (1, 150, 150, 3)


def preprocess_for_tl(pil_img: Image.Image) -> np.ndarray:

    img = pil_img.resize(TL_IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = mobilenet_preprocess(arr)           # [-1, 1]
    arr = np.asarray(arr)  # Ensure it's a numpy array
    return cast(np.ndarray, np.expand_dims(arr, axis=0))        # (1, 224, 224, 3)


def build_result(probs: np.ndarray) -> dict:
    idx        = int(np.argmax(probs))
    label      = CLASSES[idx]
    confidence = float(probs[idx]) * 100
    per_class  = [
        {
            "class":       cls,
            "probability": round(float(p) * 100, 1),
            "is_top":      i == idx,
        }
        for i, (cls, p) in enumerate(zip(CLASSES, probs))
    ]
    return {
        "prediction": label,
        "confidence": round(confidence, 1),
        "results":    per_class,
    }



@app.route("/")
def index():
    # Pass availability flags to the template
    scratch_ready = SCRATCH_MODEL.exists()
    tl_ready      = TL_MODEL.exists()
    return render_template("index.html",
                           scratch_ready=scratch_ready,
                           tl_ready=tl_ready)


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        response = {}

        # -- Scratch CNN prediction ------------------------------------------
        if SCRATCH_MODEL.exists():
            arr   = preprocess_for_scratch(pil_img)
            probs = get_scratch_model().predict(arr)[0]
            response["scratch"] = build_result(probs)
        else:
            response["scratch"] = {"error": "Model not trained yet. Run train.py."}

        # -- Transfer learning model prediction ------------------------------
        if TL_MODEL.exists():
            arr   = preprocess_for_tl(pil_img)
            probs = get_tl_model().predict(arr)[0]
            response["transfer"] = build_result(probs)
        else:
            response["transfer"] = {"error": "Model not trained yet. Run train_transfer.py."}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Pre-loading models ...")
    try:
        get_scratch_model()
        print("  [OK] Scratch CNN loaded")
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")

    try:
        get_tl_model()
        print("  [OK] Transfer learning model loaded")
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")

    print("Starting Flask server ...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, host="0.0.0.0", port=5000)
