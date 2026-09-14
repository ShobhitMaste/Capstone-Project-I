"""Run Flower Federated Learning experiment (simulation mode).

Usage:
    python experiments/run_federated.py --rounds 10 --skew 0.8 --method fedavg --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedtrap.config import get_class_names
from fedtrap.model import (
    create_mobilenetv2,
    freeze_backbone,
    get_trainable_params,
    set_trainable_params,
)
from fedtrap.dataset import InsectDataset, get_transforms
from experiments.partition_non_iid import create_label_skew
from federated.fedavg import weighted_fedavg, measure_parameter_bytes


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_local(model, loader, epochs, lr, device, method="fedavg",
                global_params=None, mu=0.01):
    """One round of local training."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # FedProx proximal term
            if method == "fedprox" and global_params is not None:
                prox = 0.0
                for p, gp in zip(
                    (p for p in model.parameters() if p.requires_grad),
                    global_params,
                ):
                    prox += ((p - gp.to(device)) ** 2).sum()
                loss += (mu / 2.0) * prox

            loss.backward()
            optimizer.step()


def evaluate_model(model, loader, device):
    """Evaluate and return accuracy."""
    from sklearn.metrics import f1_score, accuracy_score

    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Federated Learning experiment")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--skew", type=float, default=0.8)
    parser.add_argument("--method", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx mu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", default="data/split_metadata.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(args.data):
        sys.exit(f"Metadata not found: {args.data}. Run prepare_dataset.py first.")

    with open(args.data, "r") as f:
        metadata = json.load(f)

    class_names = get_class_names()
    num_classes = len(class_names)

    train_paths = metadata["train"]["images"]
    train_labels = metadata["train"]["labels"]
    test_paths = metadata["test"]["images"]
    test_labels = metadata["test"]["labels"]

    # Create test loader
    test_ds = InsectDataset(test_paths, test_labels, get_transforms(train=False))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    # Partition
    labels_arr = np.array(train_labels)
    partition = create_label_skew(labels_arr, num_clients=2, skew=args.skew, seed=args.seed)

    # Create per-client loaders
    client_loaders = []
    for ci in range(2):
        indices = partition[ci]
        paths = [train_paths[i] for i in indices]
        labels = [train_labels[i] for i in indices]
        ds = InsectDataset(paths, labels, get_transforms(train=True))
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        client_loaders.append((loader, len(indices)))

    # Global model
    global_model = create_mobilenetv2(num_classes=num_classes)
    freeze_backbone(global_model)

    total_comm = 0
    round_metrics = []

    print(f"Starting {args.method.upper()} simulation: {args.rounds} rounds, skew={args.skew}")

    for rnd in range(1, args.rounds + 1):
        global_params = get_trainable_params(global_model)
        global_tensors = [torch.tensor(p) for p in global_params]

        client_updates = []
        for ci, (loader, n_samples) in enumerate(client_loaders):
            local_model = create_mobilenetv2(num_classes=num_classes)
            freeze_backbone(local_model)
            set_trainable_params(local_model, global_params)

            train_local(
                local_model, loader, args.epochs, args.lr, device,
                method=args.method, global_params=global_tensors, mu=args.mu,
            )

            updated = get_trainable_params(local_model)
            total_comm += measure_parameter_bytes(updated)
            client_updates.append((updated, n_samples))

        # Aggregate
        aggregated = weighted_fedavg(client_updates)
        set_trainable_params(global_model, aggregated)
        total_comm += measure_parameter_bytes(aggregated) * 2  # broadcast

        # Evaluate
        acc, f1 = evaluate_model(global_model, test_loader, device)
        print(f"  Round {rnd}/{args.rounds}: Acc={acc:.4f}, F1={f1:.4f}, Comm={total_comm/1024:.1f} KB")

        round_metrics.append({
            "round": rnd,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "cumulative_bytes": total_comm,
        })

    # Save results
    results = {
        "method": args.method,
        "rounds": args.rounds,
        "skew": args.skew,
        "final_accuracy": round(acc, 4),
        "final_f1": round(f1, 4),
        "total_communication_bytes": total_comm,
        "per_round": round_metrics,
    }

    with open("results/federated_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save global model
    torch.save(global_model.state_dict(), f"models/federated_global_r{args.rounds}.pt")
    print(f"\nFL complete. Final Acc={acc:.4f}, F1={f1:.4f}, Comm={total_comm/1024:.1f} KB")
    print("Results saved to results/federated_results.json")


if __name__ == "__main__":
    main()
