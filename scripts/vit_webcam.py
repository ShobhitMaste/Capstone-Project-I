"""
FedTrap — HuggingFace ViT Insect Detection (Live Webcam)
Uses: Mustafa5645344/insect-detection-vit (Vision Transformer)

Usage:
    python scripts/vit_webcam.py
    python scripts/vit_webcam.py --source "path/to/image.jpg"
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedtrap.camera import MotionDetector

def main():
    parser = argparse.ArgumentParser(description="ViT Insect Detection — Webcam or Image")
    parser.add_argument("--source", type=str, default="webcam",
                        help="'webcam' for live feed, or path to an image/folder")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Minimum confidence to display prediction")
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────
    local_model_dir = Path(__file__).resolve().parent.parent / "models" / "DownloadedFromNet"
    print(f"Loading ViT model from: {local_model_dir}")

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained(str(local_model_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    # Get label mapping — normalize keys to int
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    num_classes = len(id2label)
    print(f"Classes ({num_classes}): {[id2label[i] for i in range(num_classes)]}")
    print()

    # ── Single image mode ───────────────────────────────────
    if args.source != "webcam":
        source_path = Path(args.source)

        # Collect image paths
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if source_path.is_file():
            image_paths = [source_path]
        elif source_path.is_dir():
            image_paths = sorted(
                p for p in source_path.rglob("*")
                if p.is_file() and p.suffix.lower() in image_exts
            )
        else:
            print(f"Error: '{args.source}' not found.")
            return

        print(f"Processing {len(image_paths)} image(s)...\n")
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                top5 = torch.topk(probs[0], min(5, num_classes))

            pred_idx = top5.indices[0].item()
            pred_label = id2label[pred_idx]
            pred_conf = top5.values[0].item()

            print(f"📷 {img_path.name}")
            print(f"   Prediction: {pred_label} ({pred_conf*100:.1f}%)")
            for i in range(min(5, num_classes)):
                idx = top5.indices[i].item()
                conf = top5.values[i].item()
                label = id2label[idx]
                bar = "█" * int(conf * 30)
                print(f"   {label:25s} {conf*100:5.1f}% {bar}")
            print()
        return

    # ── Webcam mode ─────────────────────────────────────────
    print("Opening webcam... Press 'q' to quit, 'c' to recalibrate background.\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    motion_det = MotionDetector(threshold=25, min_area=500)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        motion, thresh, contours = motion_det.detect(frame)

        if motion:
            # Find largest contour to crop the insect
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Add a small 20px padding around the insect
            pad = 20
            y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
            cropped_insect = frame[y1:y2, x1:x2]

            # Convert BGR (OpenCV) → RGB (PIL)
            rgb = cv2.cvtColor(cropped_insect, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Inference
            inputs = processor(images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_idx = probs[0].argmax().item()
                pred_conf = probs[0][pred_idx].item()
                top3 = torch.topk(probs[0], min(3, num_classes))

            pred_label = id2label[pred_idx]
        else:
            pred_conf = 0.0
            pred_label = "No motion"
            top3 = None

        # FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Draw on frame
        if pred_conf >= args.threshold:
            color = (0, 255, 0)  # Green
            text = f"{pred_label}: {pred_conf*100:.1f}%"
        else:
            color = (128, 128, 128)
            text = f"Uncertain ({pred_conf*100:.1f}%)"

        cv2.putText(display_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show top-3 predictions as small text
        if top3 is not None:
            for i in range(min(3, num_classes)):
                idx = top3.indices[i].item()
                conf = top3.values[i].item()
                label = id2label[idx]
                cv2.putText(display_frame, f"{i+1}. {label}: {conf*100:.1f}%",
                            (10, 120 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 200, 200), 1)

        cv2.imshow("ViT Insect Detection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Recalibrating background...")
            motion_det.update_background(frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

if __name__ == "__main__":
    main()
