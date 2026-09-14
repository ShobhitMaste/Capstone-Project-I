import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedtrap.inference import InsectClassifier

def main():
    parser = argparse.ArgumentParser(description="Test model on a directory of classified images")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to testing directory")
    parser.add_argument("--model", type=str, default="models/classifier_best.pt", help="Path to model file")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Error: Testing directory '{test_dir}' not found.")
        return

    print("Loading classifier...")
    classifier = InsectClassifier(model_path=args.model, config_path=args.config)
    
    # Valid classes according to config
    valid_classes = set(classifier.class_names)
    
    results = defaultdict(lambda: {"correct": 0, "total": 0, "wrong": []})
    total_correct = 0
    total_images = 0

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_folder in sorted(test_dir.iterdir()):
        if not class_folder.is_dir():
            continue
            
        true_class = class_folder.name.lower()
        if true_class not in valid_classes:
            print(f"Warning: Folder '{class_folder.name}' does not match any known class. Skipping.")
            continue
            
        print(f"\nProcessing '{true_class}'...")
        
        for img_path in sorted(class_folder.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in image_exts:
                pred = classifier.predict(str(img_path))
                pred_class = pred['class_name']
                
                results[true_class]["total"] += 1
                total_images += 1
                
                if pred_class == true_class:
                    results[true_class]["correct"] += 1
                    total_correct += 1
                else:
                    results[true_class]["wrong"].append(f"{img_path.name} -> {pred_class} ({pred['confidence']:.2f})")

    if total_images == 0:
        print("No images found in the testing directory.")
        return

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    print(f"{'Class':<15} | {'Total':<6} | {'Correct':<7} | {'Accuracy':<8}")
    print("-" * 15 + "-+-" + "-" * 6 + "-+-" + "-" * 7 + "-+-" + "-" * 8)
    
    for cls in sorted(results.keys()):
        stats = results[cls]
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        print(f"{cls.capitalize():<15} | {stats['total']:<6} | {stats['correct']:<7} | {acc:>6.1f}%")
        
    overall_acc = (total_correct / total_images) * 100 if total_images > 0 else 0
    print("-" * 15 + "-+-" + "-" * 6 + "-+-" + "-" * 7 + "-+-" + "-" * 8)
    print(f"{'OVERALL':<15} | {total_images:<6} | {total_correct:<7} | {overall_acc:>6.2f}%")
    
    print("\n--- Mismatches ---")
    mismatches_found = False
    for cls in sorted(results.keys()):
        wrong_list = results[cls]["wrong"]
        if wrong_list:
            mismatches_found = True
            print(f"\n{cls.capitalize()} misclassified as:")
            for item in wrong_list[:10]: # Print up to 10 wrong per class to avoid spam
                print(f"  - {item}")
            if len(wrong_list) > 10:
                print(f"  ... and {len(wrong_list) - 10} more.")
                
    if not mismatches_found:
        print("None! Perfect prediction.")

if __name__ == "__main__":
    main()
