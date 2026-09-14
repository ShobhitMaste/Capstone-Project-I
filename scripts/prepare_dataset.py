import os
import sys
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def get_class_names(data_dir):
    classes = []
    for d in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, d)):
            classes.append(d)
    return sorted(classes)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--data", type=str, default="data/dataset", help="Dataset directory")
    parser.add_argument("--output", type=str, default="data/split_metadata.json", help="Output metadata file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data)
    out_file = Path(args.output)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = get_class_names(data_dir)
    print(f"Discovered classes: {classes}")
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    valid_images = []
    labels = []
    corrupted_count = 0
    small_count = 0

    class_counts = defaultdict(int)

    for cls in classes:
        cls_dir = data_dir / cls
        for file in cls_dir.rglob("*"):
            if not file.is_file():
                continue
            
            try:
                with Image.open(file) as img:
                    img.verify()
                
                # We have to reopen to check size because verify() might not reliably load size for all formats
                with Image.open(file) as img:
                    width, height = img.size
                    if width < 32 or height < 32:
                        small_count += 1
                        os.remove(file)
                        continue
                
                valid_images.append(str(file))
                labels.append(class_to_idx[cls])
                class_counts[cls] += 1
                
            except Exception as e:
                corrupted_count += 1
                os.remove(file)

    print(f"Removed {corrupted_count} corrupted images and {small_count} too small (<32x32) images.")

    # Check for empty or small classes
    for cls in classes:
        count = class_counts[cls]
        if count == 0:
            print(f"ERROR: Class '{cls}' has 0 images.")
        elif count < 50:
            print(f"WARNING: Class '{cls}' has < 50 images ({count}).")

    # Stratified split: 70 / 15 / 15
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        valid_images, labels, test_size=0.3, stratify=labels, random_state=args.seed
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=args.seed
    )

    metadata = {
        "train": {"images": train_paths, "labels": train_labels},
        "val": {"images": val_paths, "labels": val_labels},
        "test": {"images": test_paths, "labels": test_labels}
    }

    with open(out_file, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(out_dir / "class_counts.json", "w") as f:
        json.dump(class_counts, f, indent=2)

    # Print summary table
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    test_counts = defaultdict(int)

    for l in metadata["train"]["labels"]: train_counts[classes[l]] += 1
    for l in metadata["val"]["labels"]: val_counts[classes[l]] += 1
    for l in metadata["test"]["labels"]: test_counts[classes[l]] += 1

    report_lines = []
    header = f"{'Class':<14}| {'Total':<6}| {'Train':<6}| {'Val':<5}| {'Test':<5}"
    sep = "-" * 14 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 6 + "+" + "-" * 5
    report_lines.append(header)
    report_lines.append(sep)

    for cls in classes:
        total = class_counts[cls]
        tr = train_counts[cls]
        va = val_counts[cls]
        te = test_counts[cls]
        report_lines.append(f"{cls:<14}| {total:<6}| {tr:<6}| {va:<5}| {te:<5}")

    report_str = "\n".join(report_lines)
    print("\n" + report_str)

    with open(out_dir / "dataset_report.txt", "w") as f:
        f.write(report_str)

if __name__ == "__main__":
    main()
