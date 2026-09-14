"""
╔══════════════════════════════════════════════════════════════╗
║  FedTrap — Cell 1: Dataset Assembly (v2 — fixed paths)       ║
║  Paste this into Kaggle Notebook Cell #1                     ║
╚══════════════════════════════════════════════════════════════╝
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

CLASS_NAMES = ["beetle", "butterfly", "grasshopper", "honeybee", "moth"]
PER_CLASS_LIMIT = 1000
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data" / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Exact paths (from Cell 0 exploration)
# ============================================================

# --- Insects Recognition ---
INSECTS_ROOT       = Path("/kaggle/input/datasets/hammaadali/insects-recognition")
INSECTS_GRASSHOPPER = INSECTS_ROOT / "Grasshopper"       # 960 images
INSECTS_BUTTERFLY   = INSECTS_ROOT / "Butterfly"         # 899 images
INSECTS_LADYBIRD    = INSECTS_ROOT / "Ladybird"          # 864 images (ladybirds = beetles!)

# --- ArTaxOr (raw images by taxonomic order) ---
ARTAXOR_ROOT       = Path("/kaggle/input/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset")
ARTAXOR_COLEOPTERA = ARTAXOR_ROOT / "ArTaxOr" / "Coleoptera"   # 2110 beetle images
ARTAXOR_LEPIDOPTERA = ARTAXOR_ROOT / "ArTaxOr" / "Lepidoptera" # 2106 moth/butterfly images

# --- BeeImage ---
BEES_ROOT          = Path("/kaggle/input/datasets/jenny18/honey-bee-annotated-images")
BEES_IMGS          = BEES_ROOT / "bee_imgs" / "bee_imgs"        # 5172 images

# --- BM100 (Butterfly & Moths 100 species) ---
BM100_ROOT         = Path("/kaggle/input/datasets/gpiosenka/butterfly-images40-species")
BM100_TRAIN        = BM100_ROOT / "train"
BM100_TEST         = BM100_ROOT / "test"
BM100_VALID        = BM100_ROOT / "valid"

# --- Agricultural Pests Image Dataset (class folders) ---
AGPESTS_ROOT       = Path("/kaggle/input/datasets/vencerlanz09/agricultural-pests-image-dataset")
AGPESTS_BEETLE     = AGPESTS_ROOT / "beetle"         # 416 images
AGPESTS_GRASSHOPPER = AGPESTS_ROOT / "grasshopper"   # 485 images
AGPESTS_MOTH       = AGPESTS_ROOT / "moth"           # 497 images
AGPESTS_BEES       = AGPESTS_ROOT / "bees"           # 500 images

# NOTE: AgroPest-12 (crop-pests-dataset) is YOLO detection format
# (images/ + labels/ .txt), NOT class folders. We skip it since
# extracting class-specific crops from YOLO format is unreliable
# without knowing which YOLO class IDs map to our insects.

# Moth keywords for BM100 heuristic
MOTH_KEYWORDS = [
    'moth', 'sphinx', 'hawkmoth', 'silkmoth', 'atlas', 'luna', 'cecropia',
    'polyphemus', 'io moth', 'rosy maple', 'emperor gum', 'tiger moth',
    'gypsy', 'codling', 'diamondback', 'armyworm', 'tussock', 'underwing',
    'geometrid', 'cinnabar', 'clearwing', 'comet', 'garden tiger',
    'giant leopard', 'bird cherry ermine', 'banded tiger', 'arcigera'
]

# ============================================================
# Helper
# ============================================================

def copy_images(src, dst, prefix, limit=None):
    """Copy images from src → dst with renamed prefix. Respects per-class limit."""
    dst.mkdir(parents=True, exist_ok=True)
    existing = len([f for f in dst.glob("*") if f.suffix.lower() in IMAGE_EXTS])
    if limit and existing >= limit:
        return 0
    n = 0
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            if limit and (existing + n) >= limit:
                break
            shutil.copy(p, dst / f"{prefix}_{n:05d}{p.suffix.lower()}")
            n += 1
    return n

# ============================================================
# Assemble Dataset
# ============================================================
print("=" * 60)
print("ASSEMBLING DATASET FROM 5 SOURCES")
print("=" * 60)

lim = PER_CLASS_LIMIT
stats = defaultdict(lambda: defaultdict(int))  # stats[class][source] = count

# ──────────────────────────────────────────────────────────
# SOURCE 1: Insects Recognition
# ──────────────────────────────────────────────────────────
print("\n📦 Source 1: Insects Recognition")

if INSECTS_GRASSHOPPER.exists():
    n = copy_images(INSECTS_GRASSHOPPER, DATA_DIR / "grasshopper", "insrec", limit=lim)
    stats["grasshopper"]["Insects Recognition"] = n
    print(f"   ✅ grasshopper: +{n}")

if INSECTS_BUTTERFLY.exists():
    n = copy_images(INSECTS_BUTTERFLY, DATA_DIR / "butterfly", "insrec", limit=lim)
    stats["butterfly"]["Insects Recognition"] = n
    print(f"   ✅ butterfly: +{n}")

if INSECTS_LADYBIRD.exists():
    n = copy_images(INSECTS_LADYBIRD, DATA_DIR / "beetle", "ladybird", limit=lim)
    stats["beetle"]["Insects Rec. (Ladybird)"] = n
    print(f"   ✅ beetle (from Ladybird — ladybirds are Coleoptera): +{n}")

# ──────────────────────────────────────────────────────────
# SOURCE 2: ArTaxOr (raw taxonomic images — no bbox needed)
# ──────────────────────────────────────────────────────────
print("\n📦 Source 2: ArTaxOr")

if ARTAXOR_COLEOPTERA.exists():
    n = copy_images(ARTAXOR_COLEOPTERA, DATA_DIR / "beetle", "artaxor_col", limit=lim)
    stats["beetle"]["ArTaxOr Coleoptera"] = n
    print(f"   ✅ beetle (Coleoptera raw images): +{n}")

# Lepidoptera is tricky — it contains BOTH moths and butterflies mixed.
# We'll use it as a butterfly top-up since we have BM100 for moth separation.
if ARTAXOR_LEPIDOPTERA.exists():
    n = copy_images(ARTAXOR_LEPIDOPTERA, DATA_DIR / "butterfly", "artaxor_lep", limit=lim)
    stats["butterfly"]["ArTaxOr Lepidoptera"] = n
    print(f"   ✅ butterfly (Lepidoptera raw images): +{n}")

# ──────────────────────────────────────────────────────────
# SOURCE 3: BeeImage
# ──────────────────────────────────────────────────────────
print("\n📦 Source 3: BeeImage")

if BEES_IMGS.exists():
    n = copy_images(BEES_IMGS, DATA_DIR / "honeybee", "beeimg", limit=lim)
    stats["honeybee"]["BeeImage"] = n
    print(f"   ✅ honeybee: +{n}")

# ──────────────────────────────────────────────────────────
# SOURCE 4: BM100 (Butterfly & Moths 100 species)
#   Uses ALL splits (train + test + valid) for maximum data.
#   Moth species identified by keyword matching.
# ──────────────────────────────────────────────────────────
print("\n📦 Source 4: BM100 (Butterfly & Moths 100 species)")

moths_found = []
butterflies_found = []
moth_total = 0
butterfly_total = 0

for split_dir in [BM100_TRAIN, BM100_TEST, BM100_VALID]:
    if not split_dir.exists():
        continue
    split_name = split_dir.name

    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        dir_name = cls_dir.name.lower()
        is_moth = any(kw in dir_name for kw in MOTH_KEYWORDS)

        if is_moth:
            n = copy_images(cls_dir, DATA_DIR / "moth",
                            f"bm100_{split_name}_{cls_dir.name}", limit=lim)
            moth_total += n
            if cls_dir.name not in moths_found:
                moths_found.append(cls_dir.name)
        else:
            n = copy_images(cls_dir, DATA_DIR / "butterfly",
                            f"bm100_{split_name}_{cls_dir.name}", limit=lim)
            butterfly_total += n
            if cls_dir.name not in butterflies_found:
                butterflies_found.append(cls_dir.name)

stats["moth"]["BM100"] = moth_total
stats["butterfly"]["BM100"] = butterfly_total
print(f"   ✅ moth: +{moth_total} ({len(moths_found)} species)")
print(f"      Species: {', '.join(moths_found)}")
print(f"   ✅ butterfly: +{butterfly_total} ({len(butterflies_found)} species)")

# ──────────────────────────────────────────────────────────
# SOURCE 5: Agricultural Pests Image Dataset
# ──────────────────────────────────────────────────────────
print("\n📦 Source 5: Agricultural Pests Image Dataset")

for src_path, cls, label in [
    (AGPESTS_BEETLE,      "beetle",      "beetle"),
    (AGPESTS_GRASSHOPPER, "grasshopper", "grasshopper"),
    (AGPESTS_MOTH,        "moth",        "moth"),
    (AGPESTS_BEES,        "honeybee",    "bees→honeybee"),
]:
    if src_path.exists():
        n = copy_images(src_path, DATA_DIR / cls, f"agpests_{src_path.name}", limit=lim)
        stats[cls]["Agricultural Pests"] = n
        print(f"   ✅ {label}: +{n}")

# ============================================================
# Per-source breakdown
# ============================================================
print("\n" + "=" * 60)
print("PER-SOURCE BREAKDOWN")
print("=" * 60)
for cls in CLASS_NAMES:
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
    json.dump({"class_to_idx": class_to_idx, "classes": classes}, f, indent=2)

# ============================================================
# Final Summary
# ============================================================
print(f"\n  Classes: {classes}")
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
print("✅ ✅ ✅  IMPORT COMPLETED  ✅ ✅ ✅")
print("=" * 60)
print(f"\n  Total valid images: {total_all}")
print(f"  Classes: {len(classes)}")
print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
print(f"\n  → Now run Cell 2 to start training!")
