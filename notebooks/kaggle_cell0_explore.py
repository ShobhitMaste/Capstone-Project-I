"""
╔══════════════════════════════════════════════════════════════╗
║  FedTrap — Cell 0: Dataset Structure Explorer                ║
║  Run this FIRST to see folder layouts before importing       ║
╚══════════════════════════════════════════════════════════════╝
"""

from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

PATHS = {
    "insects":    Path("/kaggle/input/datasets/hammaadali/insects-recognition"),
    "artaxor":    Path("/kaggle/input/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset"),
    "bees":       Path("/kaggle/input/datasets/jenny18/honey-bee-annotated-images"),
    "bm100":      Path("/kaggle/input/datasets/gpiosenka/butterfly-images40-species"),
    "agropest":   Path("/kaggle/input/datasets/rupankarmajumdar/crop-pests-dataset"),
    "agpests":    Path("/kaggle/input/datasets/vencerlanz09/agricultural-pests-image-dataset"),
}

def count_images(path):
    """Count image files in a directory (non-recursive)."""
    return sum(1 for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)

def count_images_recursive(path):
    """Count image files in a directory (recursive)."""
    return sum(1 for f in path.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS)

def count_other_files(path):
    """Count non-image files (json, xml, txt, csv, etc.)."""
    return sum(1 for f in path.iterdir() if f.is_file() and f.suffix.lower() not in IMAGE_EXTS)

def explore(root, max_depth=3, indent=0):
    """Print directory tree with image counts."""
    if not root.exists():
        print(f"{'  ' * indent}❌ PATH DOES NOT EXIST")
        return

    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    files_here = list(root.iterdir())
    img_count = sum(1 for f in files_here if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    other_count = sum(1 for f in files_here if f.is_file() and f.suffix.lower() not in IMAGE_EXTS)

    if img_count > 0 or other_count > 0:
        parts = []
        if img_count > 0:
            parts.append(f"{img_count} images")
        if other_count > 0:
            # Show what types of non-image files
            other_exts = set(f.suffix.lower() for f in files_here if f.is_file() and f.suffix.lower() not in IMAGE_EXTS)
            parts.append(f"{other_count} other [{', '.join(sorted(other_exts))}]")
        print(f"{'  ' * indent}📁 {root.name}/  ({', '.join(parts)})")
    else:
        print(f"{'  ' * indent}📁 {root.name}/")

    if indent < max_depth:
        for d in dirs[:50]:  # cap at 50 subdirs to avoid massive output
            explore(d, max_depth, indent + 1)
        if len(dirs) > 50:
            print(f"{'  ' * (indent+1)}... and {len(dirs) - 50} more folders")

# ============================================================
for name, path in PATHS.items():
    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print(f"PATH: {path}")
    print("=" * 70)

    if not path.exists():
        print("❌ NOT FOUND")
        continue

    total_imgs = count_images_recursive(path)
    print(f"Total images (recursive): {total_imgs}")
    print()
    explore(path, max_depth=3)

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE — copy this output and share it")
print("=" * 70)
