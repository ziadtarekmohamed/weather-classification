import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_dataset():
    # Define paths
    raw_dataset_dir = Path(r"c:\Users\h\Downloads\dataset\dataset")
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    classes = ["cloudy", "foggy", "rainy", "shine", "sunrise"]

    if not raw_dataset_dir.exists():
        print(f"Error: Raw dataset directory not found at {raw_dataset_dir}")
        print("Please ensure the dataset exists before running this script.")
        return

    # Recreate the data directories
    if data_dir.exists():
        print(f"Removing existing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    for c in classes:
        (train_dir / c).mkdir(parents=True, exist_ok=True)
        (val_dir / c).mkdir(parents=True, exist_ok=True)

    print("Splitting dataset into train and val...")
    
    total_train = 0
    total_val = 0

    for c in classes:
        class_dir = raw_dataset_dir / c
        if not class_dir.exists():
            print(f"Warning: Class directory {class_dir} does not exist.")
            continue
            
        images = [f for f in class_dir.iterdir() if f.is_file()]
        
        if len(images) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
            
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        for img in train_imgs:
            shutil.copy(img, train_dir / c / img.name)
            
        for img in val_imgs:
            shutil.copy(img, val_dir / c / img.name)
            
        print(f"  {c}: {len(train_imgs)} train, {len(val_imgs)} val images")
        
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        
    print(f"Dataset preparation complete. Total: {total_train} train, {total_val} val.")

if __name__ == "__main__":
    prepare_dataset()
