"""
╔══════════════════════════════════════════════════════════════╗
║  FedTrap — Cell 1: Dataset Assembly (EfficientNet-B0 v3)     ║
║  Sources: AgroPest-12 + Insects Recognition                  ║
║  Paste this into Kaggle Notebook Cell #1                     ║
╚══════════════════════════════════════════════════════════════╝

Merging strategy:
  - AgroPest-12: 12 classes used directly (ants, bees, beetle, caterpillar,
    earthworms, earwig, grasshopper, moth, slug, snail, wasp, weevil)
  - Insects Recognition:
      • Butterfly   → new class "butterfly"
      • Dragonfly   → new class "dragonfly"
      • Grasshopper → merged into existing "grasshopper"
      • Ladybird    → merged into "beetle" (ladybirds are Coleoptera)
      • Mosquito    → new class "mosquito"

Final: 15 classes, ~9,900 images
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================

SEED = 42
random.seed(SEED)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data" / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset Paths
# ============================================================

# --- Agricultural Pests Image Dataset (12 classes) ---
AGPESTS_ROOT = Path("/kaggle/input/datasets/vencerlanz09/agricultural-pests-image-dataset")

# --- Insects Recognition (5 classes) ---
INSECTS_ROOT = Path("/kaggle/input/datasets/hammaadali/insects-recognition")

# ============================================================
# Merge Mapping
# ============================================================
# Each entry: (source_path, target_class_name, description)

AGPESTS_MAPPING = [
    ("ants",         "ants",         "ants"),
    ("bees",         "bees",         "bees"),
    ("beetle",       "beetle",       "beetle"),
    ("catterpillar", "caterpillar",  "caterpillar"),       # fix typo in dataset
    ("earthworms",   "earthworms",   "earthworms"),
    ("earwig",       "earwig",       "earwig"),
    ("grasshopper",  "grasshopper",  "grasshopper"),
    ("moth",         "moth",         "moth"),
    ("slug",         "slug",         "slug"),
    ("snail",        "snail",        "snail"),
    ("wasp",         "wasp",         "wasp"),
    ("weevil",       "weevil",       "weevil"),
]

INSECTS_MAPPING = [
    ("Butterfly",    "butterfly",    "butterfly"),
    ("Dragonfly",    "dragonfly",    "dragonfly"),
    ("Grasshopper",  "grasshopper",  "grasshopper (merge)"),   # merge with agpests
    ("Ladybird",     "beetle",       "ladybird → beetle"),     # ladybirds are Coleoptera
    ("Mosquito",     "mosquito",     "mosquito"),
]

# ============================================================
# Helper
# ============================================================

def copy_images(src, dst, prefix):
    """Copy all images from src → dst with renamed prefix to avoid collisions."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    existing = len(list(dst.glob("*")))
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            shutil.copy(p, dst / f"{prefix}_{existing + n:05d}{p.suffix.lower()}")
            n += 1
    return n

# ============================================================
# Assemble Dataset
# ============================================================
print("=" * 60)
print("ASSEMBLING DATASET FROM 2 SOURCES")
print("=" * 60)

stats = defaultdict(lambda: defaultdict(int))  # stats[class][source] = count

# ──────────────────────────────────────────────────────────
# SOURCE 1: Agricultural Pests Image Dataset
# ──────────────────────────────────────────────────────────
print("\n📦 Source 1: Agricultural Pests Image Dataset (AgroPest-12)")

for src_folder, target_class, desc in AGPESTS_MAPPING:
    src_path = AGPESTS_ROOT / src_folder
    if src_path.exists():
        n = copy_images(src_path, DATA_DIR / target_class, f"agpest_{src_folder}")
        stats[target_class]["AgroPest-12"] += n
        print(f"   ✅ {desc}: +{n} images → {target_class}/")
    else:
        print(f"   ❌ {src_folder} not found!")

# ──────────────────────────────────────────────────────────
# SOURCE 2: Insects Recognition
# ──────────────────────────────────────────────────────────
print("\n📦 Source 2: Insects Recognition")

for src_folder, target_class, desc in INSECTS_MAPPING:
    src_path = INSECTS_ROOT / src_folder
    if src_path.exists():
        n = copy_images(src_path, DATA_DIR / target_class, f"insrec_{src_folder.lower()}")
        stats[target_class]["Insects Recognition"] += n
        print(f"   ✅ {desc}: +{n} images → {target_class}/")
    else:
        print(f"   ❌ {src_folder} not found!")

# ============================================================
# Per-source breakdown
# ============================================================
print("\n" + "=" * 60)
print("PER-SOURCE BREAKDOWN")
print("=" * 60)

all_classes = sorted(stats.keys())
for cls in all_classes:
    print(f"\n  {cls.upper()}:")
    cls_total = 0
    for source, count in stats[cls].items():
        print(f"    {source:30s} → {count:5d} images")
        cls_total += count
    print(f"    {'SUBTOTAL':30s} → {cls_total:5d} images")

# ============================================================
# Image Validation & Cleaning
# ============================================================
print("\n" + "=" * 60)
print("VALIDATING & CLEANING IMAGES")
print("=" * 60)

classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
NUM_CLASSES = len(classes)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

valid_images = []
labels = []
corrupted_count = 0
small_count = 0
class_counts = defaultdict(int)

for cls in classes:
    cls_dir = DATA_DIR / cls
    for file in cls_dir.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(file) as img:
                img.verify()
            with Image.open(file) as img:
                w, h = img.size
                if w < 32 or h < 32:
                    small_count += 1
                    continue
            valid_images.append(str(file))
            labels.append(class_to_idx[cls])
            class_counts[cls] += 1
        except Exception:
            corrupted_count += 1

print(f"  Removed {corrupted_count} corrupted images")
print(f"  Removed {small_count} images smaller than 32x32")

# ============================================================
# Stratified Split: 70 / 15 / 15
# ============================================================
print("\n" + "=" * 60)
print("CREATING TRAIN / VAL / TEST SPLIT (70/15/15)")
print("=" * 60)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    valid_images, labels, test_size=0.3, stratify=labels, random_state=SEED
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=SEED
)

split_metadata = {
    "train": {"images": train_paths, "labels": train_labels},
    "val":   {"images": val_paths,   "labels": val_labels},
    "test":  {"images": test_paths,  "labels": test_labels}
}

meta_path = WORK_DIR / "data" / "split_metadata.json"
meta_path.parent.mkdir(parents=True, exist_ok=True)
with open(meta_path, "w") as f:
    json.dump(split_metadata, f, indent=2)

with open(WORK_DIR / "data" / "class_mapping.json", "w") as f:
    json.dump({"class_to_idx": class_to_idx, "classes": classes, "num_classes": NUM_CLASSES}, f, indent=2)

# ============================================================
# Final Summary
# ============================================================
print(f"\n  Classes ({NUM_CLASSES}): {classes}")
print(f"  Mapping: {class_to_idx}")
print(f"\n  {'Class':15s} | {'Total':>6s} | {'Train':>6s} | {'Val':>5s} | {'Test':>5s}")
print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")

total_all = 0
for cls in classes:
    idx = class_to_idx[cls]
    tr = sum(1 for l in train_labels if l == idx)
    va = sum(1 for l in val_labels if l == idx)
    te = sum(1 for l in test_labels if l == idx)
    total = class_counts[cls]
    total_all += total
    print(f"  {cls:15s} | {total:6d} | {tr:6d} | {va:5d} | {te:5d}")

print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")
print(f"  {'TOTAL':15s} | {total_all:6d} | {len(train_paths):6d} | {len(val_paths):5d} | {len(test_paths):5d}")

print(f"\n  Saved: {meta_path}")

# ============================================================
print("\n" + "=" * 60)
print("✅ ✅ ✅  DATASET ASSEMBLY COMPLETED  ✅ ✅ ✅")
print("=" * 60)
print(f"\n  Total valid images: {total_all}")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
print(f"\n  → Now run Cell 2 to start EfficientNet-B0 training!")
