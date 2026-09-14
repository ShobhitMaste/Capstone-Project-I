import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedtrap.config import load_labels_json
from fedtrap.model import load_model
from fedtrap.dataset import create_dataloaders

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    acc = correct / total
    return acc, all_preds, all_labels

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Classifier")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--data", type=str, default="data/split_metadata.json", help="Path to split metadata JSON")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split to evaluate on")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_to_class = load_labels_json()
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    print(f"Loading model from {args.model}")
    model = load_model(args.model, num_classes=num_classes)
    model.to(device)
    
    dataloaders = create_dataloaders(args.data, batch_size=32, img_size=args.img_size)
    if args.split not in dataloaders:
        print(f"Error: split '{args.split}' not found in metadata.")
        return
        
    dataloader = dataloaders[args.split]
    
    print(f"Running evaluation on {args.split} split...")
    acc, preds, labels = evaluate(model, dataloader, device)
    
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(labels, preds)
    
    plot_confusion_matrix(cm, class_names, str(out_dir / f"eval_{args.split}_confusion_matrix.png"))
    
    with open(out_dir / f"eval_{args.split}_classification_report.txt", "w") as f:
        f.write(report)
        
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    metrics = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": {}
    }
    for i, c in enumerate(class_names):
        metrics["per_class"][c] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i]
        }
        
    with open(out_dir / f"eval_{args.split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nEvaluation Results ({args.split}):")
    print(f"Accuracy: {acc:.4f}")
    print(report)

if __name__ == "__main__":
    main()
