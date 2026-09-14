"""Run LOCAL vs CENTRALIZED vs FEDERATED comparison.

Usage:
    python experiments/run_comparison.py --rounds 10 --skew 0.8 --epochs 5 --seed 42

Generates:
    results/local_results.json
    results/centralized_results.json
    results/federated_results.json
    results/comparison.csv
    results/comparison.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedtrap.config import get_class_names, get_config
from fedtrap.model import create_mobilenetv2, freeze_backbone, unfreeze_top_layers, save_model
from fedtrap.dataset import InsectDataset, get_transforms
from experiments.partition_non_iid import create_label_skew, visualize_partition
from federated.fedavg import weighted_fedavg, measure_parameter_bytes
from fedtrap.model import get_trainable_params, set_trainable_params


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, epochs, lr, device):
    """Train model locally."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model, return accuracy, macro-F1, per-class F1."""
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return accuracy, macro_f1, per_class_f1.tolist()


def main():
    parser = argparse.ArgumentParser(description="LOCAL vs CENTRALIZED vs FL comparison")
    parser.add_argument("--rounds", type=int, default=10, help="FL rounds")
    parser.add_argument("--skew", type=float, default=0.8, help="Non-IID skew")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", default="data/split_metadata.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    # Load split metadata
    if not os.path.exists(args.data):
        sys.exit(f"Split metadata not found: {args.data}. Run prepare_dataset.py first.")

    with open(args.data, "r") as f:
        metadata = json.load(f)

    class_names = get_class_names()
    num_classes = len(class_names)

    # Extract train and test data
    train_paths = metadata["train"]["images"]
    train_labels = metadata["train"]["labels"]
    test_paths = metadata["test"]["images"]
    test_labels = metadata["test"]["labels"]

    # Create test loader
    test_dataset = InsectDataset(test_paths, test_labels, get_transforms(train=False))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Partition training data for non-IID
    labels_arr = np.array(train_labels)
    partition = create_label_skew(labels_arr, num_clients=2, skew=args.skew, seed=args.seed)
    visualize_partition(partition, labels_arr, class_names, "results/partition_distribution.png")

    print(f"Partition sizes: Client A = {len(partition[0])}, Client B = {len(partition[1])}")

    total_train_epochs = args.rounds * args.epochs
    results = []

    # =============================================
    # 1. LOCAL-ONLY (each client trains in isolation)
    # =============================================
    print("\n=== LOCAL-ONLY ===")
    for ci, client_name in enumerate(["Client A", "Client B"]):
        set_seed(args.seed)
        indices = partition[ci]
        client_paths = [train_paths[i] for i in indices]
        client_labels = [train_labels[i] for i in indices]
        ds = InsectDataset(client_paths, client_labels, get_transforms(train=True))
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

        model = create_mobilenetv2(num_classes=num_classes)
        freeze_backbone(model)
        model = train_model(model, loader, total_train_epochs, args.lr, device)

        acc, f1, pc_f1 = evaluate_model(model, test_loader, device)
        print(f"  {client_name}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}")

        results.append({
            "Method": f"Local ({client_name})",
            "Accuracy": round(acc, 4),
            "Macro-F1": round(f1, 4),
            "Communication (bytes)": 0,
            "Rounds": 0,
            **{f"F1_{class_names[k]}": round(pc_f1[k], 4) for k in range(num_classes)},
        })

    save_results("results/local_results.json", results[-2:])

    # =============================================
    # 2. CENTRALIZED (all data pooled)
    # =============================================
    print("\n=== CENTRALIZED ===")
    set_seed(args.seed)
    all_ds = InsectDataset(train_paths, train_labels, get_transforms(train=True))
    all_loader = torch.utils.data.DataLoader(all_ds, batch_size=args.batch_size, shuffle=True)

    model = create_mobilenetv2(num_classes=num_classes)
    freeze_backbone(model)
    model = train_model(model, all_loader, total_train_epochs, args.lr, device)

    acc, f1, pc_f1 = evaluate_model(model, test_loader, device)
    print(f"  Centralized: Accuracy={acc:.4f}, Macro-F1={f1:.4f}")

    results.append({
        "Method": "Centralized",
        "Accuracy": round(acc, 4),
        "Macro-F1": round(f1, 4),
        "Communication (bytes)": 0,
        "Rounds": 0,
        **{f"F1_{class_names[k]}": round(pc_f1[k], 4) for k in range(num_classes)},
    })
    save_results("results/centralized_results.json", [results[-1]])

    # =============================================
    # 3. FEDERATED (FedAvg simulation)
    # =============================================
    print("\n=== FEDERATED (FedAvg) ===")
    set_seed(args.seed)

    # Create per-client loaders
    client_loaders = []
    for ci in range(2):
        indices = partition[ci]
        paths = [train_paths[i] for i in indices]
        labels = [train_labels[i] for i in indices]
        ds = InsectDataset(paths, labels, get_transforms(train=True))
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        client_loaders.append((loader, len(indices)))

    # Initialize global model
    global_model = create_mobilenetv2(num_classes=num_classes)
    freeze_backbone(global_model)

    total_comm_bytes = 0
    fl_metrics = []

    for rnd in range(1, args.rounds + 1):
        client_updates = []

        for ci, (loader, n_samples) in enumerate(client_loaders):
            # Clone global model for local training
            local_model = create_mobilenetv2(num_classes=num_classes)
            freeze_backbone(local_model)
            global_params = get_trainable_params(global_model)
            set_trainable_params(local_model, global_params)

            # Local training
            local_model = train_model(local_model, loader, args.epochs, args.lr, device)
            updated_params = get_trainable_params(local_model)

            # Measure communication
            bytes_up = measure_parameter_bytes(updated_params)
            total_comm_bytes += bytes_up

            client_updates.append((updated_params, n_samples))

        # Aggregate
        aggregated = weighted_fedavg(client_updates)
        set_trainable_params(global_model, aggregated)

        # Broadcast cost
        bytes_down = measure_parameter_bytes(aggregated) * 2
        total_comm_bytes += bytes_down

        # Evaluate global model
        acc, f1, pc_f1 = evaluate_model(global_model, test_loader, device)
        print(f"  Round {rnd}/{args.rounds}: Accuracy={acc:.4f}, F1={f1:.4f}, Comm={total_comm_bytes/1024:.1f} KB")

        fl_metrics.append({
            "round": rnd,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "cumulative_bytes": total_comm_bytes,
        })

    results.append({
        "Method": "Federated (FedAvg)",
        "Accuracy": round(acc, 4),
        "Macro-F1": round(f1, 4),
        "Communication (bytes)": total_comm_bytes,
        "Rounds": args.rounds,
        **{f"F1_{class_names[k]}": round(pc_f1[k], 4) for k in range(num_classes)},
    })
    save_results("results/federated_results.json", {"final": results[-1], "per_round": fl_metrics})

    # =============================================
    # Comparison outputs
    # =============================================
    df = pd.DataFrame(results)
    df.to_csv("results/comparison.csv", index=False)
    print(f"\n{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}")

    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = df["Method"].tolist()
    accs = df["Accuracy"].tolist()
    f1s = df["Macro-F1"].tolist()

    colors = ["#4f8cff", "#4f8cff", "#36d399", "#f87272"][:len(methods)]
    ax1.barh(methods, accs, color=colors)
    ax1.set_xlabel("Accuracy")
    ax1.set_title("Method Comparison — Accuracy")
    ax1.set_xlim(0, 1)

    ax2.barh(methods, f1s, color=colors)
    ax2.set_xlabel("Macro-F1")
    ax2.set_title("Method Comparison — Macro-F1")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=150)
    plt.close()
    print("Comparison saved to results/comparison.csv and results/comparison.png")


def save_results(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
