"""
FedTrap — YOLO11n AgroPest-12 Object Detection (Live Webcam)
Uses: models/YOLO11n/train/weights/best.pt

Usage:
    python scripts/yolo11_webcam.py
"""

import sys
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO11 AgroPest-12 Detection")
    parser.add_argument(
        "--model", 
        type=str, 
        default=str(Path(__file__).resolve().parent.parent / "models" / "YOLO11n" / "train" / "weights" / "best.pt"),
        help="Path to YOLO11n best.pt"
    )
    parser.add_argument("--threshold", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading YOLO11 model from: {model_path}")
    model = YOLO(str(model_path))
    
    print("\nClasses detected by this model:")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")

    print("\nStarting live webcam Object Detection... Press 'q' inside the video window to quit.")
    
    # We use Ultralytics' built-in streaming which natively handles OpenCV boxes, FPS, and rendering!
    results = model.predict(source=0, show=True, conf=args.threshold, stream=True)
    
    # Iterate over the stream to keep it running
    for r in results:
        pass
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
