"""Download the source datasets via kagglehub and assemble data/dataset/.

Setup (once):  pip install kagglehub pillow   +  ~/.kaggle/kaggle.json token.
Run:           python scripts/download_datasets.py [--out data/dataset]

Automated:  grasshopper + butterfly (Insects Recognition), beetle (ArTaxOr
Coleoptera, bbox-cropped), honeybee (BeeImage). Printed as MANUAL: the
moth/butterfly species split and IP102 adult-only picks — see DATASETS.md.
Dataset folder layouts occasionally change on Kaggle; if a section fails, the
script prints the downloaded tree so you can adjust the paths at the top.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DATASETS = {
    "insects": "hammaadali/insects-recognition",
    "artaxor": "mistag/arthropod-taxonomy-orders-object-detection-dataset",
    "bees": "jenny18/honey-bee-annotated-images",
    "bm100": "gpiosenka/100-butterfly-species",
    # "ip102": "rtlmhjbn/ip02-dataset",   # large (~3 GB) — enable when needed
}


def tree(root: Path, depth: int = 2) -> None:
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if len(rel.parts) <= depth and p.is_dir():
            n = sum(1 for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            print(f"   {rel}  ({n} images)")


def copy_images(src: Path, dst: Path, prefix: str, limit: int | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    existing = len(list(dst.glob("*")))
    n = 0
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            if limit and (existing + n) >= limit:
                break
            shutil.copy(p, dst / f"{prefix}_{n:05d}{p.suffix.lower()}")
            n += 1
    return n


def find_dir(root: Path, keyword: str) -> Path | None:
    """First directory whose name contains keyword (case-insensitive)."""
    for p in sorted(root.rglob("*")):
        if p.is_dir() and keyword.lower() in p.name.lower():
            return p
    return None


def crop_artaxor_coleoptera(art_root: Path, dst: Path, limit: int = 600) -> int:
    """ArTaxOr: images + per-image JSON annotations with normalized bboxes."""
    from PIL import Image

    col = find_dir(art_root, "coleoptera")
    if col is None:
        print("  !! Coleoptera folder not found — tree below; adjust manually")
        tree(art_root)
        return 0
    ann_dir = find_dir(col, "annotation") or col
    dst.mkdir(parents=True, exist_ok=True)
    existing = len(list(dst.glob("*")))
    n = 0
    for jf in sorted(ann_dir.rglob("*.json")):
        try:
            if existing + n >= limit:
                return n
            meta = json.loads(jf.read_text(encoding="utf8"))
            asset = meta.get("asset", {})
            img_name = asset.get("name") or asset.get("path", "").split("/")[-1]
            img_path = next(col.rglob(img_name), None) if img_name else None
            if img_path is None:
                continue
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            for region in meta.get("regions", []):
                bb = region.get("boundingBox") or {}
                if not bb:
                    continue
                left, top = bb.get("left", 0), bb.get("top", 0)
                w, h = bb.get("width", 0), bb.get("height", 0)
                # ArTaxOr stores fractions of image size
                box = (int(left * W), int(top * H),
                       int((left + w) * W), int((top + h) * H))
                if box[2] - box[0] < 32 or box[3] - box[1] < 32:
                    continue
                img.crop(box).save(dst / f"artaxor_{n:05d}.jpg", quality=90)
                n += 1
                if existing + n >= limit:
                    return n
        except Exception:
            continue
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--per-class-limit", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out)

    try:
        import kagglehub
    except ImportError:
        sys.exit("pip install kagglehub  (and put your kaggle.json token in ~/.kaggle/)")

    paths: dict[str, Path] = {}
    for key, ds in DATASETS.items():
        if key == "ip102":
            continue  # comment this line out and uncomment in DATASETS to download IP102
        print(f"downloading {ds} ...")
        paths[key] = Path(kagglehub.dataset_download(ds))
        print(f"  -> {paths[key]}")

    lim = args.per_class_limit

    # grasshopper + butterfly from Insects Recognition
    for kw, cls in (("grasshopper", "grasshopper"), ("butterfly", "butterfly")):
        src = find_dir(paths["insects"], kw)
        if src:
            n = copy_images(src, out / cls, "insrec", limit=lim)
            print(f"{cls}: +{n} from Insects Recognition")
        else:
            print(f"!! {kw} folder not found in Insects Recognition:")
            tree(paths["insects"])

    # beetle from ArTaxOr Coleoptera (bbox crops)
    n = crop_artaxor_coleoptera(paths["artaxor"], out / "beetle", limit=lim)
    print(f"beetle: +{n} from ArTaxOr Coleoptera (bbox-cropped)")

    # honeybee from BeeImage
    n = copy_images(paths["bees"], out / "honeybee", "beeimg", limit=lim)
    print(f"honeybee: +{n} from BeeImage")
    
    # Moth and Butterfly from Butterfly & Moths 100 species
    moth_keywords = [
        'moth', 'sphinx', 'hawkmoth', 'silkmoth', 'atlas', 'luna', 'cecropia', 
        'polyphemus', 'io moth', 'rosy maple', 'emperor', 'tiger moth', 'gypsy', 
        'codling', 'diamondback', 'armyworm', 'tussock', 'underwing', 'geometrid'
    ]
    bm100_root = paths.get("bm100")
    if bm100_root:
        train_dir = find_dir(bm100_root, "train") or bm100_root
        moths_found = []
        butterflies_found = []
        moth_count = 0
        butterfly_count = 0
        
        for cls_dir in sorted(train_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            dir_name = cls_dir.name.lower()
            
            is_moth = any(kw in dir_name for kw in moth_keywords)
            if is_moth:
                moths_found.append(cls_dir.name)
                moth_count += copy_images(cls_dir, out / "moth", f"bm100_{cls_dir.name}", limit=lim)
            else:
                butterflies_found.append(cls_dir.name)
                butterfly_count += copy_images(cls_dir, out / "butterfly", f"bm100_{cls_dir.name}", limit=lim)
        
        print(f"\n--- Butterfly & Moths 100 Species Report ---")
        print(f"Moth species detected ({len(moths_found)}): {', '.join(moths_found[:10])}{'...' if len(moths_found) > 10 else ''}")
        print(f"Butterfly species detected ({len(butterflies_found)}): {', '.join(butterflies_found[:10])}{'...' if len(butterflies_found) > 10 else ''}")
        print(f"Added {moth_count} moth images and {butterfly_count} butterfly images.")

    # IP102 Top-up (Commented out by default)
    # instructions to enable: Uncomment "ip102" in DATASETS above, and uncomment this section.
    # ip102_root = paths.get("ip102")
    # if ip102_root:
    #     print("Processing IP102 top-up...")
    #     ip102_map = {"grasshopper": ["locust"], "beetle": ["chafer", "weevil"], "moth": ["moth", "armyworm", "borer"]}
    #     for cls_name, kws in ip102_map.items():
    #         for kw in kws:
    #             src = find_dir(ip102_root, kw)
    #             if src:
    #                 n = copy_images(src, out / cls_name, f"ip102_{kw}", limit=lim)
    #                 print(f"{cls_name}: +{n} from IP102 ({kw})")

    print("\nMANUAL steps remaining (see DATASETS.md):")
    print(" - later: add YOUR trap-camera shots to every class (highest value)")
    print("\nnow run:  python scripts/prepare_dataset.py --data", out)


if __name__ == "__main__":
    main()
