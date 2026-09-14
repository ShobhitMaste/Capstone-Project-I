"""
╔══════════════════════════════════════════════════════════════╗
║  FedTrap — Cell 2: Training + Evaluation + FL Comparison     ║
║  Paste this into Kaggle Notebook Cell #2                     ║
║  (Run Cell 1 first!)                                         ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import random
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 0. Configuration
# ============================================================

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
FREEZE_EPOCHS = 10       # Phase 1: train classifier head only
FINETUNE_EPOCHS = 20     # Phase 2: unfreeze top-4 backbone blocks
TOTAL_EPOCHS = FREEZE_EPOCHS + FINETUNE_EPOCHS
LR = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# FL experiment config
FL_ROUNDS = 10
FL_LOCAL_EPOCHS = 3
FL_SKEW = 0.8

WORK_DIR = Path("/kaggle/working")
MODELS_DIR = WORK_DIR / "models"
RESULTS_DIR = WORK_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 1. Load split metadata from Cell 1
# ============================================================
print("\n" + "=" * 60)
print("LOADING DATASET FROM CELL 1")
print("=" * 60)

meta_path = WORK_DIR / "data" / "split_metadata.json"
mapping_path = WORK_DIR / "data" / "class_mapping.json"

assert meta_path.exists(), "❌ split_metadata.json not found! Run Cell 1 first."
assert mapping_path.exists(), "❌ class_mapping.json not found! Run Cell 1 first."

with open(meta_path) as f:
    split_metadata = json.load(f)
with open(mapping_path) as f:
    class_info = json.load(f)

classes = class_info["classes"]
class_to_idx = class_info["class_to_idx"]
NUM_CLASSES = len(classes)

train_paths = split_metadata["train"]["images"]
train_labels = split_metadata["train"]["labels"]
val_paths = split_metadata["val"]["images"]
val_labels = split_metadata["val"]["labels"]
test_paths = split_metadata["test"]["images"]
test_labels = split_metadata["test"]["labels"]

print(f"  Classes ({NUM_CLASSES}): {classes}")
print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

# ============================================================
# 2. Model + Dataset Classes
# ============================================================

def get_transforms_fn(train=True, img_size=224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class InsectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def create_mobilenetv2(num_classes=5, pretrained=True):
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    return model

def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False

def unfreeze_top_layers(model, n=4):
    blocks = list(model.features.children())
    start_idx = max(0, len(blocks) - n)
    for i in range(start_idx, len(blocks)):
        for param in blocks[i].parameters():
            param.requires_grad = True

# ============================================================
# 3. Dataloaders
# ============================================================

train_ds = InsectDataset(train_paths, train_labels, get_transforms_fn(True, IMG_SIZE))
val_ds   = InsectDataset(val_paths, val_labels, get_transforms_fn(False, IMG_SIZE))
test_ds  = InsectDataset(test_paths, test_labels, get_transforms_fn(False, IMG_SIZE))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"  DataLoaders: train={len(train_loader)} batches, val={len(val_loader)}, test={len(test_loader)}")

# ============================================================
# 4. Training Functions
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels_batch in loader:
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels_batch in loader:
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels

# ============================================================
# 5. Training
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: TRAINING")
print("=" * 60)

model = create_mobilenetv2(num_classes=NUM_CLASSES, pretrained=True).to(device)

# Class weights for imbalanced data
label_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
class_weights = 1.0 / (label_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print(f"  Class weights: {dict(zip(classes, [f'{w:.3f}' for w in class_weights.tolist()]))}")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# --- Phase 1: Frozen backbone ---
print("\n--- Phase 1: Training Classifier Head (backbone frozen) ---")
freeze_backbone(model)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {trainable:,}")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)

best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(1, FREEZE_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion)
    scheduler.step()

    train_losses.append(tr_loss)
    val_losses.append(val_loss)
    train_accs.append(tr_acc)
    val_accs.append(val_acc)

    print(f"  Epoch [{epoch:2d}/{TOTAL_EPOCHS}] Phase 1 | "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), str(MODELS_DIR / "classifier_best.pt"))
        print(f"    → Best model saved (val_acc={val_acc:.4f})")

# --- Phase 2: Fine-tune top layers ---
print(f"\n--- Phase 2: Fine-tuning Top-4 Backbone Blocks ---")
unfreeze_top_layers(model, n=4)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {trainable:,}")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR / 10, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

for epoch in range(FREEZE_EPOCHS + 1, TOTAL_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion)
    scheduler.step()

    train_losses.append(tr_loss)
    val_losses.append(val_loss)
    train_accs.append(tr_acc)
    val_accs.append(val_acc)

    print(f"  Epoch [{epoch:2d}/{TOTAL_EPOCHS}] Phase 2 | "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), str(MODELS_DIR / "classifier_best.pt"))
        print(f"    → Best model saved (val_acc={val_acc:.4f})")

# Save final model
torch.save(model.state_dict(), str(MODELS_DIR / "classifier.pt"))
print(f"\n✅ Training complete. Best val accuracy: {best_val_acc:.4f}")

# ============================================================
# 6. Evaluation & Plots
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: EVALUATION")
print("=" * 60)

# Load best checkpoint
model.load_state_dict(torch.load(str(MODELS_DIR / "classifier_best.pt"), map_location=device))
test_loss, test_acc, test_preds, test_true = evaluate_model(model, test_loader, criterion)

report = classification_report(test_true, test_preds, target_names=classes, digits=4)
cm = confusion_matrix(test_true, test_preds)

print(f"\n  Test Accuracy: {test_acc:.4f}")
print(report)

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(train_losses) + 1)

ax1.plot(epochs_range, train_losses, 'b-o', markersize=3, label='Train Loss')
ax1.plot(epochs_range, val_losses, 'r-o', markersize=3, label='Val Loss')
ax1.axvline(x=FREEZE_EPOCHS, color='gray', linestyle='--', alpha=0.7, label='Unfreeze')
ax1.set_title('Training & Validation Loss', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accs, 'b-o', markersize=3, label='Train Accuracy')
ax2.plot(epochs_range, val_accs, 'r-o', markersize=3, label='Val Accuracy')
ax2.axvline(x=FREEZE_EPOCHS, color='gray', linestyle='--', alpha=0.7, label='Unfreeze')
ax2.set_title('Training & Validation Accuracy', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "training_curves.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (Test Set)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save metrics
precision, recall, f1, _ = precision_recall_fscore_support(test_true, test_preds, average=None, zero_division=0)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(test_true, test_preds, average='macro', zero_division=0)

metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "macro_precision": float(macro_p),
    "macro_recall": float(macro_r),
    "macro_f1": float(macro_f1),
    "per_class": {}
}
for i, c in enumerate(classes):
    metrics["per_class"][c] = {
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i])
    }

with open(RESULTS_DIR / "training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(report)

print("  Saved: training_curves.png, confusion_matrix.png, training_metrics.json")

# ============================================================
# 7. Federated Learning Comparison (Simulated)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: FEDERATED COMPARISON EXPERIMENT")
print("=" * 60)

# Non-IID partition (Dirichlet)
def dirichlet_partition(labels, num_clients=2, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    unique_labels = np.unique(labels_arr)
    client_indices = [[] for _ in range(num_clients)]
    for lbl in unique_labels:
        idx = np.where(labels_arr == lbl)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        splits = (proportions * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()
        start = 0
        for c in range(num_clients):
            client_indices[c].extend(idx[start:start+splits[c]].tolist())
            start += splits[c]
    return client_indices

alpha = 1.0 - FL_SKEW
partitions = dirichlet_partition(train_labels, num_clients=2, alpha=max(alpha, 0.1), seed=SEED)

# Plot partition distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for c_idx, (indices, ax) in enumerate(zip(partitions, axes)):
    c_labels = [train_labels[i] for i in indices]
    counts = np.bincount(c_labels, minlength=NUM_CLASSES)
    ax.bar(classes, counts, color=['#e74c3c', '#e67e22', '#9b59b6', '#f1c40f', '#3498db'])
    ax.set_title(f'Client {"A" if c_idx==0 else "B"} ({len(indices)} samples)', fontsize=13)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
plt.suptitle(f'Non-IID Partition (Dirichlet α={max(alpha, 0.1):.2f})', fontsize=14)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "partition_distribution.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f"  Partition sizes: Client A={len(partitions[0])}, Client B={len(partitions[1])}")

# Helper functions
def get_params(model):
    return [p.detach().cpu().numpy().copy() for p in model.parameters() if p.requires_grad]

def set_params(model, params):
    trainable = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        for p, p_new in zip(trainable, params):
            p.copy_(torch.tensor(p_new, dtype=p.dtype, device=p.device))

def measure_bytes(params):
    return sum(p.nbytes for p in params)

# --- Local-Only Training ---
print("\n=== LOCAL-ONLY ===")
local_results = {}
for c_idx, indices in enumerate(partitions):
    client_name = f"Client {'A' if c_idx==0 else 'B'}"
    m = create_mobilenetv2(NUM_CLASSES, pretrained=True).to(device)
    freeze_backbone(m)
    unfreeze_top_layers(m, n=4)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR/10, weight_decay=WEIGHT_DECAY)

    c_paths = [train_paths[i] for i in indices]
    c_labels_list = [train_labels[i] for i in indices]
    c_ds = InsectDataset(c_paths, c_labels_list, get_transforms_fn(True, IMG_SIZE))
    c_loader = DataLoader(c_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    crit = nn.CrossEntropyLoss()
    for ep in range(FL_LOCAL_EPOCHS * FL_ROUNDS):
        train_one_epoch(m, c_loader, crit, opt)

    _, acc, preds, trues = evaluate_model(m, test_loader, crit)
    mf1 = f1_score(trues, preds, average='macro', zero_division=0)
    per_f1 = f1_score(trues, preds, average=None, zero_division=0)
    local_results[client_name] = {"accuracy": acc, "macro_f1": mf1, "per_class_f1": per_f1.tolist()}
    print(f"  {client_name}: Accuracy={acc:.4f}, Macro-F1={mf1:.4f}")
    del m, opt, c_ds, c_loader
    torch.cuda.empty_cache()

# --- Centralized Training ---
print("\n=== CENTRALIZED ===")
m_cent = create_mobilenetv2(NUM_CLASSES, pretrained=True).to(device)
freeze_backbone(m_cent)
unfreeze_top_layers(m_cent, n=4)
opt_cent = optim.AdamW(filter(lambda p: p.requires_grad, m_cent.parameters()), lr=LR/10, weight_decay=WEIGHT_DECAY)
crit_cent = nn.CrossEntropyLoss()

for ep in range(FL_LOCAL_EPOCHS * FL_ROUNDS):
    train_one_epoch(m_cent, train_loader, crit_cent, opt_cent)

_, cent_acc, cent_preds, cent_trues = evaluate_model(m_cent, test_loader, crit_cent)
cent_f1 = f1_score(cent_trues, cent_preds, average='macro', zero_division=0)
cent_per_f1 = f1_score(cent_trues, cent_preds, average=None, zero_division=0)
print(f"  Centralized: Accuracy={cent_acc:.4f}, Macro-F1={cent_f1:.4f}")
del m_cent, opt_cent
torch.cuda.empty_cache()

# --- Federated (FedAvg) ---
print("\n=== FEDERATED (FedAvg) ===")
global_model = create_mobilenetv2(NUM_CLASSES, pretrained=True).to(device)
freeze_backbone(global_model)
unfreeze_top_layers(global_model, n=4)

total_comm = 0.0
fed_round_metrics = []

for rnd in range(1, FL_ROUNDS + 1):
    global_params = get_params(global_model)
    client_updates = []
    client_sizes = []

    for c_idx, indices in enumerate(partitions):
        local_model = create_mobilenetv2(NUM_CLASSES, pretrained=False).to(device)
        freeze_backbone(local_model)
        unfreeze_top_layers(local_model, n=4)
        set_params(local_model, global_params)

        c_paths_fl = [train_paths[i] for i in indices]
        c_labels_fl = [train_labels[i] for i in indices]
        c_ds_fl = InsectDataset(c_paths_fl, c_labels_fl, get_transforms_fn(True, IMG_SIZE))
        c_loader_fl = DataLoader(c_ds_fl, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

        opt_fl = optim.AdamW(filter(lambda p: p.requires_grad, local_model.parameters()),
                             lr=LR/10, weight_decay=WEIGHT_DECAY)
        crit_fl = nn.CrossEntropyLoss()

        for ep in range(FL_LOCAL_EPOCHS):
            train_one_epoch(local_model, c_loader_fl, crit_fl, opt_fl)

        updated_params = get_params(local_model)
        client_updates.append(updated_params)
        client_sizes.append(len(indices))
        total_comm += measure_bytes(updated_params)
        del local_model, opt_fl, c_ds_fl, c_loader_fl

    # FedAvg aggregation
    total_samples = sum(client_sizes)
    new_global = []
    for param_idx in range(len(client_updates[0])):
        weighted = sum(
            client_updates[c][param_idx] * (client_sizes[c] / total_samples)
            for c in range(len(client_updates))
        )
        new_global.append(weighted)

    set_params(global_model, new_global)

    _, rnd_acc, rnd_preds, rnd_trues = evaluate_model(global_model, test_loader, nn.CrossEntropyLoss())
    rnd_f1 = f1_score(rnd_trues, rnd_preds, average='macro', zero_division=0)
    fed_round_metrics.append({"round": rnd, "accuracy": rnd_acc, "f1": rnd_f1, "comm_kb": total_comm/1024})
    print(f"  Round {rnd:2d}/{FL_ROUNDS}: Accuracy={rnd_acc:.4f}, F1={rnd_f1:.4f}, Comm={total_comm/1024:.1f} KB")
    torch.cuda.empty_cache()

# Save federated model
torch.save(global_model.state_dict(), str(MODELS_DIR / "federated_best.pt"))

_, fed_acc, fed_preds, fed_trues = evaluate_model(global_model, test_loader, nn.CrossEntropyLoss())
fed_f1 = f1_score(fed_trues, fed_preds, average='macro', zero_division=0)
fed_per_f1 = f1_score(fed_trues, fed_preds, average=None, zero_division=0)

# ============================================================
# 8. Comparison Table & Plots
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

rows = []
for client_name, res in local_results.items():
    row = {
        "Method": f"Local ({client_name})",
        "Accuracy": res["accuracy"],
        "Macro-F1": res["macro_f1"],
        "Communication (bytes)": 0,
        "Rounds": 0,
    }
    for i, c in enumerate(classes):
        row[f"F1_{c}"] = res["per_class_f1"][i]
    rows.append(row)

row_cent = {
    "Method": "Centralized",
    "Accuracy": cent_acc,
    "Macro-F1": cent_f1,
    "Communication (bytes)": 0,
    "Rounds": 0,
}
for i, c in enumerate(classes):
    row_cent[f"F1_{c}"] = float(cent_per_f1[i])
rows.append(row_cent)

row_fed = {
    "Method": "Federated (FedAvg)",
    "Accuracy": fed_acc,
    "Macro-F1": fed_f1,
    "Communication (bytes)": int(total_comm),
    "Rounds": FL_ROUNDS,
}
for i, c in enumerate(classes):
    row_fed[f"F1_{c}"] = float(fed_per_f1[i])
rows.append(row_fed)

# Print table
header = f"  {'Method':>20s}  {'Accuracy':>8s}  {'Macro-F1':>8s}  {'Comm (KB)':>10s}  {'Rounds':>6s}"
print(header)
print("  " + "-" * (len(header) - 2))
for r in rows:
    print(f"  {r['Method']:>20s}  {r['Accuracy']:>8.4f}  {r['Macro-F1']:>8.4f}  "
          f"{r['Communication (bytes)']/1024:>10.1f}  {r['Rounds']:>6d}")

# Save CSV
fieldnames = list(rows[0].keys())
with open(RESULTS_DIR / "comparison.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Comparison bar chart
methods = [r["Method"] for r in rows]
accs = [r["Accuracy"] for r in rows]
f1s = [r["Macro-F1"] for r in rows]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(methods))

ax1.bar(x, accs, color=colors[:len(methods)], edgecolor='white', linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
ax1.set_ylabel('Accuracy')
ax1.set_title('Test Accuracy Comparison', fontsize=13)
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(accs):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

ax2.bar(x, f1s, color=colors[:len(methods)], edgecolor='white', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
ax2.set_ylabel('Macro F1-Score')
ax2.set_title('Macro F1 Comparison', fontsize=13)
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(f1s):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle(f'FedTrap: Local vs Centralized vs Federated (Non-IID α={max(alpha,0.1):.2f})', fontsize=14)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "comparison.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# FL convergence plot
if fed_round_metrics:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rounds_x = [m["round"] for m in fed_round_metrics]

    ax1.plot(rounds_x, [m["accuracy"] for m in fed_round_metrics], 'b-o', markersize=5, label='Federated')
    ax1.axhline(y=cent_acc, color='r', linestyle='--', linewidth=2, label=f'Centralized ({cent_acc:.3f})')
    for cname, res in local_results.items():
        ax1.axhline(y=res["accuracy"], color='gray', linestyle=':', alpha=0.5,
                     label=f'{cname} ({res["accuracy"]:.3f})')
    ax1.set_xlabel('FL Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Federated Learning Convergence — Accuracy')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(rounds_x, [m["f1"] for m in fed_round_metrics], 'g-o', markersize=5, label='Federated')
    ax2.axhline(y=cent_f1, color='r', linestyle='--', linewidth=2, label=f'Centralized ({cent_f1:.3f})')
    for cname, res in local_results.items():
        ax2.axhline(y=res["macro_f1"], color='gray', linestyle=':', alpha=0.5,
                     label=f'{cname} ({res["macro_f1"]:.3f})')
    ax2.set_xlabel('FL Round')
    ax2.set_ylabel('Macro F1')
    ax2.set_title('Federated Learning Convergence — F1')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "fl_convergence.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ============================================================
# 9. Summary
# ============================================================
print("\n" + "=" * 60)
print("✅ ✅ ✅  ALL DONE  ✅ ✅ ✅")
print("=" * 60)
print("\nFiles saved in /kaggle/working/:")
for d in [MODELS_DIR, RESULTS_DIR]:
    for f in sorted(d.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(WORK_DIR)}  ({size_kb:.1f} KB)")

print(f"\n→ Download 'models/classifier_best.pt' → place in project files/models/")
print(f"→ Download 'models/federated_best.pt'  → federated variant")
print(f"→ Download 'results/' folder            → for your project report")
