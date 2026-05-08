import sys
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image

BASE_DIR   = Path(r"C:\Users\h\Documents\Weatherimg")
MODEL_PATH = BASE_DIR / "weather_model.h5"
IMG_SIZE   = (150, 150)
CLASSES    = ["cloudy", "foggy", "rainy", "shine", "sunrise"]

if not MODEL_PATH.exists():
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run train.py first.")
    sys.exit(1)

model = keras.models.load_model(str(MODEL_PATH))

def predict(img_path: str) -> None:
    path = Path(img_path)
    if not path.exists():
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    img = keras_image.load_img(str(path), target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = CLASSES[idx]
    conf  = probs[idx] * 100

    print("\n" + "=" * 45)
    print(f"  Image  : {path.name}")
    print(f"  Result : {label.upper()}  ({conf:.1f}%)")
    print("=" * 45)
    print("\n  All class probabilities:")
    for i, (cls, p) in enumerate(zip(CLASSES, probs)):
        bar = "#" * int(p * 30)
        marker = " <--" if i == idx else ""
        print(f"    {cls:<10} {p*100:5.1f}%  {bar}{marker}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
    predict(sys.argv[1])
