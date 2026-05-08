"""
app.py
------
Flask web application for weather image classification.

Run:
  python app.py
"""

import os
import io
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template, jsonify
from PIL import Image

# -- Config ------------------------------------------------------------------
BASE_DIR   = Path(r"C:\Users\h\Documents\Weatherimg")
MODEL_PATH = BASE_DIR / "weather_model.h5"
IMG_SIZE   = (150, 150)
CLASSES    = ["cloudy", "foggy", "rainy", "shine", "sunrise"]

# -- Flask app ---------------------------------------------------------------
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "static"))

# -- Load model once ---------------------------------------------------------
model = None

def get_model():
    global model
    if model is None:
        from tensorflow import keras
        model = keras.models.load_model(str(MODEL_PATH))
    return model

# -- Routes ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        m = get_model()
        probs = m.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = CLASSES[idx]
        confidence = float(probs[idx]) * 100

        results = []
        for i, (cls, p) in enumerate(zip(CLASSES, probs)):
            results.append({
                "class": cls,
                "probability": round(float(p) * 100, 1),
                "is_top": i == idx,
            })

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 1),
            "results": results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading model...")
    get_model()
    print("Model loaded. Starting server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, host="0.0.0.0", port=5000)
