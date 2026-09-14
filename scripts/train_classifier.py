import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedtrap.config import save_labels_json
from fedtrap.model import create_mobilenetv2, freeze_backbone, unfreeze_top_layers, save_model
from fedtrap.dataset import create_dataloaders

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_class_mapping(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    mapping = {}
    for item_path, label in zip(metadata['train']['images'], metadata['train']['labels']):
        if label not in mapping:
            # Extract class name from path: data/dataset/<class_name>/image.jpg
            class_name = Path(item_path).parent.name
            mapping[label] = class_name
    return mapping

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

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
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Classifier")
    parser.add_argument("--data", type=str, default="data/split_metadata.json", help="Path to split metadata JSON")
    parser.add_argument("--epochs", type=int, default=15, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--output", type=str, default="models/", help="Output directory for models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Number of epochs to keep backbone frozen")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get labels mapping
    idx_to_class = get_class_mapping(args.data)
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    save_labels_json(idx_to_class)
    print(f"Discovered {num_classes} classes: {class_names}")

    # Dataloaders
    dataloaders = create_dataloaders(args.data, batch_size=args.batch_size, img_size=args.img_size)
    
    # Model Setup
    model = create_mobilenetv2(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1 setup
    freeze_backbone(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.freeze_epochs)
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("--- Phase 1: Training Classifier Head ---")
    for epoch in range(1, args.freeze_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, dataloaders['val'], criterion, device)
        scheduler.step()
        
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch}/{args.epochs}] Phase 1 | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, str(out_dir / "classifier_best.pt"))
            
    print("--- Phase 2: Fine-tuning Top Layers ---")
    unfreeze_top_layers(model, n=4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.freeze_epochs))
    
    for epoch in range(args.freeze_epochs + 1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, dataloaders['val'], criterion, device)
        scheduler.step()
        
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch}/{args.epochs}] Phase 2 | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, str(out_dir / "classifier_best.pt"))
            
    save_model(model, str(out_dir / "classifier.pt"))
    print("Training complete. Models saved.")
    
    # Plot curves
    plot_curves(train_losses, val_losses, train_accs, val_accs, str(results_dir / "training_curves.png"))
    
    # Evaluation on Test Split
    print("--- Evaluating on Test Split ---")
    model.load_state_dict(torch.load(str(out_dir / "classifier_best.pt")))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, dataloaders['test'], criterion, device)
    
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(test_labels, test_preds)
    
    plot_confusion_matrix(cm, class_names, str(results_dir / "confusion_matrix.png"))
    
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report)
        
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average=None)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')
    
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
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
        
    with open(results_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(report)

if __name__ == "__main__":
    main()
