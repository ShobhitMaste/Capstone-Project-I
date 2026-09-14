"""Non-IID data partitioning for federated learning experiments.

Supports Dirichlet-based partitioning and explicit label-skew control.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =====================================================================
# Core partitioning functions
# =====================================================================


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int = 2,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[np.ndarray]:
    """Partition data indices using Dirichlet distribution.

    Lower alpha → more non-IID.  Higher alpha → closer to IID.

    Returns:
        List of numpy arrays, one per client, containing data indices.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    num_classes = len(np.unique(labels))

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        class_idx = np.where(labels == k)[0]
        rng.shuffle(class_idx)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to split points
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        parts = np.split(class_idx, splits)

        for i in range(num_clients):
            client_indices[i].extend(parts[i].tolist())

    # Shuffle each client's indices
    result = []
    for indices in client_indices:
        arr = np.array(indices)
        rng.shuffle(arr)
        result.append(arr)

    return result


def create_label_skew(
    labels: np.ndarray,
    num_clients: int = 2,
    skew: float = 0.8,
    seed: int = 42,
) -> List[np.ndarray]:
    """Controlled label-skew partitioning for 2 clients.

    skew=0.0  → IID (50/50 split for every class)
    skew=1.0  → extreme non-IID (client A dominates first half of classes,
                client B dominates the rest)

    For >2 clients, falls back to Dirichlet with computed alpha.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    num_classes = len(np.unique(labels))

    if num_clients != 2:
        alpha = max(0.01, 10.0 * (1.0 - skew))
        return dirichlet_partition(labels, num_clients, alpha, seed)

    client_indices: List[List[int]] = [[], []]
    class_split = num_classes // 2  # first half vs second half

    for k in range(num_classes):
        class_idx = np.where(labels == k)[0]
        rng.shuffle(class_idx)
        n = len(class_idx)

        # Client 0 gets high proportion of classes 0..class_split-1
        # Client 0 gets low  proportion of classes class_split..end
        if k < class_split:
            prop_a = 0.5 + (skew / 2.0)
        else:
            prop_a = 0.5 - (skew / 2.0)

        split_point = int(n * prop_a)
        # Ensure both clients get at least 1 sample when possible
        split_point = max(1, min(split_point, n - 1)) if n > 1 else split_point

        client_indices[0].extend(class_idx[:split_point].tolist())
        client_indices[1].extend(class_idx[split_point:].tolist())

    result = []
    for indices in client_indices:
        arr = np.array(indices)
        rng.shuffle(arr)
        result.append(arr)

    return result


# =====================================================================
# Visualization & saving
# =====================================================================


def visualize_partition(
    partition: List[np.ndarray],
    labels: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Bar chart of class distribution per client."""
    import matplotlib.pyplot as plt

    labels = np.asarray(labels)
    num_clients = len(partition)
    num_classes = len(class_names)

    counts = np.zeros((num_clients, num_classes))
    for i, indices in enumerate(partition):
        client_labels = labels[indices]
        for k in range(num_classes):
            counts[i, k] = np.sum(client_labels == k)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_clients)
    bottom = np.zeros(num_clients)

    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))
    for k in range(num_classes):
        ax.bar(x, counts[:, k], bottom=bottom, label=class_names[k], color=colors[k])
        bottom += counts[:, k]

    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {i+1}" for i in range(num_clients)])
    ax.set_ylabel("Number of Samples")
    ax.set_title("Non-IID Data Partition")
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Partition plot saved to {save_path}")


def save_partition(
    partition: List[np.ndarray],
    image_paths: List[str],
    labels: List[int],
    save_dir: str,
) -> None:
    """Save partition metadata to JSON."""
    os.makedirs(save_dir, exist_ok=True)
    data = {}
    for i, indices in enumerate(partition):
        data[f"client_{i}"] = {
            "indices": indices.tolist(),
            "image_paths": [image_paths[idx] for idx in indices],
            "labels": [labels[idx] for idx in indices],
        }

    path = os.path.join(save_dir, "partition_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Partition metadata saved to {path}")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    from fedtrap.config import get_class_names

    parser = argparse.ArgumentParser(description="Non-IID data partitioning")
    parser.add_argument("--data", default="data/split_metadata.json")
    parser.add_argument("--skew", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--method", choices=["dirichlet", "skew"], default="skew")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"Data file {args.data} not found. Run prepare_dataset.py first.")

    with open(args.data, "r") as f:
        data = json.load(f)

    if "train" not in data:
        sys.exit("No 'train' split in metadata.")

    image_paths = data["train"]["images"]
    labels_list = data["train"]["labels"]
    labels_arr = np.array(labels_list)
    class_names = get_class_names()

    if args.method == "dirichlet":
        alpha = max(0.01, 10.0 * (1.0 - args.skew))
        partition = dirichlet_partition(labels_arr, args.clients, alpha, args.seed)
    else:
        partition = create_label_skew(labels_arr, args.clients, args.skew, args.seed)

    visualize_partition(partition, labels_arr, class_names, "results/partition_distribution.png")
    save_partition(partition, image_paths, labels_list, "data/")
    print(f"Partitioned {len(labels_list)} samples across {args.clients} clients (skew={args.skew})")
