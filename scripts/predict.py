import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedtrap.inference import InsectClassifier
from fedtrap.decision_policy import get_decision

def main():
    parser = argparse.ArgumentParser(description="Single Image Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/classifier_best.pt", help="Path to model file")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        return

    classifier = InsectClassifier(model_path=args.model, config_path=args.config)
    
    pred = classifier.predict(args.image)
    
    print(f"Predicted: {pred['class_name']}")
    print(f"Confidence: {pred['confidence']:.2f}")
    print("\nAll probabilities:")
    for name, prob in pred['probabilities'].items():
        print(f"  {name:<12}: {prob:.2f}")
        
    print()
    decision = get_decision(pred['class_name'], pred['confidence'])
    
    # Apply override threshold from args
    if pred['confidence'] < args.threshold:
        print(f"Category: UNKNOWN")
        print(f"Action: NO_ACTION")
    else:
        print(f"Category: {decision['category']}")
        print(f"Action: {decision['action']}")

if __name__ == "__main__":
    main()
