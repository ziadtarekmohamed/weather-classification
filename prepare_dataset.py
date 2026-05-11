
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

RAW_DATASET_DIR = Path(r"D:\University\Image processing\dataset\dataset")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"

CLASSES = ["cloudy", "foggy", "rainy", "shine", "sunrise"]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15  

RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



def is_valid_image(path: Path) -> bool:

    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    try:
        with Image.open(path) as img:
            img.verify() 
        return True
    except Exception:
        return False


def print_class_distribution(class_counts: dict) -> None:
 
    total = sum(class_counts.values())
    print("\n" + "=" * 55)
    print("  CLASS DISTRIBUTION (valid images only)")
    print("=" * 55)
    print(f"  {'Class':<12} {'Count':>6}  {'Share':>6}")
    print("-" * 55)

    counts = list(class_counts.values())
    for cls, cnt in class_counts.items():
        pct = cnt / total * 100 if total else 0
        print(f"  {cls:<12} {cnt:>6}  {pct:>5.1f}%")

    print("=" * 55)
    print(f"  Total valid images: {total}")

    if counts and max(counts) > 2 * min(counts):
        print("\n  ⚠  CLASS IMBALANCE DETECTED:")
        print("     The largest class has more than 2× the samples")
        print("     of the smallest class.")
        print("     Mitigation applied in train.py:")
        print("       • class_weight='balanced' passed to model.fit()")
        print("       • Data augmentation on the training split")
    else:
        print("\n  ✓  Class distribution is reasonably balanced.")
    print()



def prepare_dataset() -> None:
    if not RAW_DATASET_DIR.exists():
        print(f"[ERROR] Raw dataset directory not found:\n  {RAW_DATASET_DIR}")
        print("Please update RAW_DATASET_DIR at the top of this script.")
        return

    print("Step 1: Scanning and validating images …")
    class_images: dict[str, list[Path]] = {}
    corrupted_count = 0

    for cls in CLASSES:
        cls_dir = RAW_DATASET_DIR / cls
        if not cls_dir.exists():
            print(f"  [WARNING] Class folder missing: {cls_dir}")
            class_images[cls] = []
            continue

        valid, invalid = [], []
        for f in sorted(cls_dir.iterdir()):
            if f.is_file():
                if is_valid_image(f):
                    valid.append(f)
                else:
                    invalid.append(f)

        if invalid:
            print(f"  [WARNING] {cls}: skipped {len(invalid)} corrupted/non-image file(s)")
            corrupted_count += len(invalid)

        class_images[cls] = valid

    print(f"  Scan complete. Corrupted/skipped files: {corrupted_count}")

    print("\nStep 2: Reporting class distribution …")
    print_class_distribution({c: len(imgs) for c, imgs in class_images.items()})

    print("Step 3: (Re)creating output directories …")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

    print("Step 4: Splitting and copying images (70 / 15 / 15) …\n")
    totals = {"train": 0, "val": 0, "test": 0}

    for cls in CLASSES:
        images = class_images[cls]
        if len(images) == 0:
            print(f"  [SKIP] {cls}: no valid images found")
            continue

        train_val_imgs, test_imgs = train_test_split(
            images,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED,
        )

        val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        train_imgs, val_imgs = train_test_split(
            train_val_imgs,
            test_size=val_fraction,
            random_state=RANDOM_SEED,
        )

        for img in train_imgs:
            shutil.copy2(img, TRAIN_DIR / cls / img.name)
        for img in val_imgs:
            shutil.copy2(img, VAL_DIR / cls / img.name)
        for img in test_imgs:
            shutil.copy2(img, TEST_DIR / cls / img.name)

        print(
            f"  {cls:<10}  train={len(train_imgs):>4}  "
            f"val={len(val_imgs):>4}  test={len(test_imgs):>4}  "
            f"total={len(images):>4}"
        )
        totals["train"] += len(train_imgs)
        totals["val"]   += len(val_imgs)
        totals["test"]  += len(test_imgs)

    grand_total = sum(totals.values())
    print(f"\n  {'TOTAL':<10}  train={totals['train']:>4}  "
          f"val={totals['val']:>4}  test={totals['test']:>4}  "
          f"total={grand_total:>4}")
    print("\nDataset preparation complete.")
    print(f"  Output directory: {DATA_DIR}")


if __name__ == "__main__":
    prepare_dataset()
