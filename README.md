# Weather Image Classifier

A deep learning system that classifies sky/weather photographs into **5 categories**:
`cloudy`, `foggy`, `rainy`, `shine`, `sunrise`.

The model is a custom Convolutional Neural Network (CNN) built entirely from scratch (no transfer learning), served through a Flask web application.

---

## Dataset

| Property | Detail |
|---|---|
| Classes | cloudy, foggy, rainy, shine, sunrise |
| Source | [Multi-class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) |
| Split | 70% Train · 15% Validation · 15% Test |
| Input size | 150 × 150 pixels, RGB |

The dataset is split **per class** so each split has the same class proportions (stratified split). Class imbalance is detected automatically and mitigated via balanced class weights during training.

---

## Project Structure

```
Weatherimg/
├── app.py                  # Flask web server
├── train.py                # CNN training pipeline
├── predict.py              # Command-line prediction tool
├── prepare_dataset.py      # Dataset splitting & validation
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── weather_model.h5        # Saved model (generated after training)
├── training_history.png    # Accuracy/loss curves (generated after training)
├── data/
│   ├── train/              # 70% of images, per class
│   ├── val/                # 15% of images, per class
│   └── test/               # 15% of images, per class (held out)
├── templates/
│   └── index.html          # Web UI
└── static/
    └── style.css           # Web UI styles
```

---

## Preprocessing

Preprocessing is implemented in `prepare_dataset.py` and applied by `train.py`.

### Dataset Preparation (`prepare_dataset.py`)
1. **Image validation** — each file is opened with Pillow and `.verify()` is called. Corrupted, truncated, or non-image files are silently skipped and reported.
2. **Extension filtering** — only `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` files are accepted.
3. **Class distribution report** — prints per-class sample counts and flags imbalance (> 2× ratio between largest and smallest class).
4. **Stratified split** — each class is split independently: 70% train · 15% val · 15% test.

### Data Augmentation (`train.py` — training set only)
| Technique | Value |
|---|---|
| Rescale (normalise) | ÷ 255 → [0, 1] |
| Rotation | ± 20° |
| Zoom | ± 20% |
| Width shift | ± 10% |
| Height shift | ± 10% |
| Horizontal flip | random |
| Brightness | [0.80, 1.20] |
| Fill mode | nearest |

Validation and test sets are **only rescaled** — never augmented — to ensure unbiased metrics.

### Inference Preprocessing (`app.py`, `predict.py`)
Images are resized to 150 × 150 and pixel values divided by 255, exactly matching the training normalisation.

---

## Model Architecture

A custom CNN built layer-by-layer — no pre-trained weights.

```
Input: (150, 150, 3)
│
├── Block 1: Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2)   → 75×75×32
├── Block 2: Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2)   → 37×37×64
├── Block 3: Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool(2×2)  → 18×18×128
├── Block 4: Conv2D(256, 3×3, ReLU) → BatchNorm → MaxPool(2×2)  →  9×9×256
│
├── Flatten
├── Dense(512, ReLU)
├── Dropout(0.5)
└── Dense(5, Softmax)   ← output: probability over 5 classes
```

**Design decisions:**
- Filter counts double per block (32 → 256): early layers detect edges/textures, later layers detect high-level patterns (cloud shapes, fog haze, etc.)
- BatchNormalization after every Conv layer stabilises gradients and speeds up convergence
- Dropout(0.5) in the classifier head is the primary overfitting guard
- Softmax output + categorical_crossentropy loss is the correct combination for multiclass classification

---

## Training

### Callbacks
| Callback | Config |
|---|---|
| EarlyStopping | monitor `val_accuracy`, patience=5, restore best weights |
| ReduceLROnPlateau | monitor `val_loss`, factor=0.5, patience=3, min_lr=1e-6 |
| ModelCheckpoint | save best `val_accuracy` only |

### Class Imbalance
Balanced class weights are computed automatically from training-set counts using `sklearn.utils.class_weight.compute_class_weight` and passed to `model.fit(class_weight=...)`.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Edit the `RAW_DATASET_DIR` path at the top of `prepare_dataset.py` to point to your downloaded dataset, then run:
```bash
python prepare_dataset.py
```

### 3. Train the Model
```bash
python train.py
```
This produces `weather_model.h5` and `training_history.png`.

### 4. Launch the Web App
```bash
python app.py
```
Open **http://localhost:5000** in your browser. Upload any sky photo to get a prediction.

### 5. Command-line Prediction
```bash
python predict.py path/to/image.jpg
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow >= 2.16.0` | Backend for Keras model execution |
| `keras >= 3.0.0` | Model building and training API |
| `numpy >= 1.23.0` | Array operations |
| `Pillow >= 9.4.0` | Image I/O and validation |
| `scikit-learn >= 1.2.0` | Dataset splitting, class weight computation |
| `matplotlib >= 3.6.0` | Training history plots |
| `flask >= 3.0.0` | Web server |
| `opencv-python >= 4.7.0` | Optional image utility |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | *(run `train.py` to populate)* |
| Test Loss | *(run `train.py` to populate)* |

Training history plot (`training_history.png`) is generated automatically after each training run.

---

## Notes

- All preprocessing is consistent between training and inference (same resize, same normalisation).
- Augmentation is applied **only** to the training set to prevent data leakage.
- The test set is held out from both training and hyperparameter tuning to give an unbiased performance estimate.
- The model is saved only when `val_accuracy` improves, preventing overfitting from extra epochs.
