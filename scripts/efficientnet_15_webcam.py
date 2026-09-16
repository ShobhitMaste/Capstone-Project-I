"""
FedTrap — EfficientNet-B0 15-Class Insect Classification (Webcam)
Uses: models/dad/classifier_best_fixed.pt

Usage:
    python scripts/efficientnet_15_webcam.py
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
from PIL import Image

# Import our custom MotionDetector
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedtrap.camera import MotionDetector

def create_efficientnet_b0(num_classes):
    """Recreate the EfficientNet-B0 structure from Kaggle Cell 2"""
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="EfficientNet-B0 15-Class Webcam Inference")
    parser.add_argument(
        "--model", 
        type=str, 
        default=str(Path(__file__).resolve().parent.parent / "models" / "dad" / "classifier_best_fixed.pt"),
        help="Path to the trained .pt model"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--source", type=str, default="0", help="'0' for webcam, or path to image")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Make sure you downloaded 'classifier_best.pt' and placed it correctly!")
        return

    # The 15 classes from our Kaggle script (alphabetical)
    class_names = [
        'ants', 'bees', 'beetle', 'butterfly', 'caterpillar', 'dragonfly', 
        'earthworms', 'earwig', 'grasshopper', 'mosquito', 'moth', 
        'slug', 'snail', 'wasp', 'weevil'
    ]
    num_classes = len(class_names)
    print(f"\nModel Classes ({num_classes}): {class_names}")

    print(f"Loading PyTorch model from: {model_path.name} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_efficientnet_b0(num_classes).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval()

    # Inference transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.source == "0" or args.source == "webcam":
        print("\nOpening webcam... Press 'q' to quit, 'c' to recalibrate background.")
        cap = cv2.VideoCapture(0)
        is_video = True
    else:
        print(f"\nProcessing image: {args.source}")
        cap = cv2.VideoCapture(args.source)
        is_video = False

    if not cap.isOpened():
        print("Error: Cannot open video/image source.")
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

        if is_video:
            motion, thresh, contours = motion_det.detect(frame)
        else:
            motion = True
            contours = [np.array([[[0, 0]], [[frame.shape[1], 0]], [[frame.shape[1], frame.shape[0]]], [[0, frame.shape[0]]]])]

        if motion:
            # Smart Cropping: Crop the insect using the largest motion contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Draw a box on the screen so the user sees what the AI is looking at!
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            pad = 20
            y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
            cropped_insect = frame[y1:y2, x1:x2]

            # Convert to PIL Image for torchvision transforms
            rgb_img = cv2.cvtColor(cropped_insect, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                
            pred_conf = probs.max().item()
            pred_idx = probs.argmax().item()
            pred_label = class_names[pred_idx]
            
            # Get top 3 indices
            top3_indices = torch.argsort(probs, descending=True)[:3].tolist()
        else:
            pred_conf = 0.0
            pred_label = "No motion"
            top3_indices = []

        # FPS calculation
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Draw UI on frame
        if pred_conf >= args.threshold and motion:
            color = (0, 255, 0)
            text = f"{pred_label}: {pred_conf*100:.1f}%"
        else:
            color = (128, 128, 128)
            text = f"Uncertain ({pred_conf*100:.1f}%)" if motion else "No motion"

        cv2.putText(display_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show top-3 predictions
        if motion:
            for i, idx in enumerate(top3_indices):
                conf = probs[idx].item()
                label = class_names[idx]
                cv2.putText(display_frame, f"{i+1}. {label}: {conf*100:.1f}%",
                            (10, 120 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("EfficientNet-B0 15-Class Classification", display_frame)
        
        if is_video:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Recalibrating background...")
                motion_det.update_background(frame)
        else:
            print("Press any key in the image window to exit...")
            cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
