"""
╔══════════════════════════════════════════════════════════════╗
║  FedTrap — Cell 3: Generate Review 3 Presentation Assets     ║
║  Paste this into Kaggle Notebook Cell #3                     ║
║  (Run this after Cell 2 finishes training)                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Create dedicated directory for Review 3 assets
REVIEW3_DIR = WORK_DIR / "results_review3"
REVIEW3_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("GENERATING REVIEW 3 PRESENTATION ASSETS")
print("=" * 60)

# 1. Dataset Distribution Chart
print("📊 1. Generating Dataset Distribution Chart...")
fig, ax = plt.subplots(figsize=(14, 6))
class_counts_arr = np.bincount(train_labels + val_labels + test_labels, minlength=NUM_CLASSES)
train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
val_counts = np.bincount(val_labels, minlength=NUM_CLASSES)
test_counts = np.bincount(test_labels, minlength=NUM_CLASSES)

x = np.arange(NUM_CLASSES)
width = 0.25
ax.bar(x - width, train_counts, width, label='Train', color='#3498db', alpha=0.85)
ax.bar(x, val_counts, width, label='Validation', color='#2ecc71', alpha=0.85)
ax.bar(x + width, test_counts, width, label='Test', color='#e74c3c', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Number of Images')
ax.set_title('Dataset Distribution Across 15 Insect Classes (Train/Val/Test Split)', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)
for i in range(NUM_CLASSES):
    ax.text(i, class_counts_arr[i] + 10, str(class_counts_arr[i]), ha='center', fontsize=7, fontweight='bold')
plt.tight_layout()
plt.savefig(str(REVIEW3_DIR / "1_dataset_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Per-Class Precision / Recall / F1 Bar Chart
print("📊 2. Generating Per-Class Performance Chart...")
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(NUM_CLASSES)
width = 0.25
ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.85)
ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.85)
ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.15)
ax.set_title(f'Per-Class Performance — EfficientNet-B0 (Overall Accuracy: {test_acc*100:.2f}%)', fontsize=13)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
for i in range(NUM_CLASSES):
    ax.text(i + width, f1[i] + 0.02, f'{f1[i]:.2f}', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig(str(REVIEW3_DIR / "2_per_class_metrics.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Normalized Confusion Matrix (Heatmap)
print("📊 3. Generating Normalized Confusion Matrix...")
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=classes, yticklabels=classes)
plt.title('Normalized Confusion Matrix — EfficientNet-B0 (% per True Class)', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(str(REVIEW3_DIR / "3_confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Model Summary Text File
print("📝 4. Generating Executive Summary...")
summary_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                FedTrap — Review 3 Summary                    ║
╚══════════════════════════════════════════════════════════════╝

Model Architecture:     EfficientNet-B0 (ImageNet-pretrained)
Input Resolution:       {IMG_SIZE} x {IMG_SIZE} x 3
Number of Classes:      {NUM_CLASSES}
Total Parameters:       {total_params:,}

Dataset:
  Total Images:          {len(train_labels) + len(val_labels) + len(test_labels)}
  Train / Val / Test:    {len(train_labels)} / {len(val_labels)} / {len(test_labels)}
  Sources:               AgroPest-12 + Insects Recognition (merged)

Results:
  Test Accuracy:         {test_acc*100:.2f}%
  Macro Precision:       {macro_p*100:.2f}%
  Macro Recall:          {macro_r*100:.2f}%
  Macro F1-Score:        {macro_f1*100:.2f}%

Federated Learning:
  Algorithm:             FedAvg (Federated Averaging)
  Clients:               2 (simulated non-IID partitions)
  Data Heterogeneity:    Dirichlet α={max(1.0 - FL_SKEW, 0.1):.2f}
"""
with open(REVIEW3_DIR / "4_executive_summary.txt", "w") as f:
    f.write(summary_text)

print("\n" + "=" * 60)
print("✅ ✅ ✅  REVIEW 3 ASSETS COMPLETED  ✅ ✅ ✅")
print("=" * 60)
print("\nFiles saved in /kaggle/working/results_review3/:")
for f in sorted(REVIEW3_DIR.glob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.1f} KB)")
print("\n→ Download the 'results_review3' folder and put these directly into your presentation!")
