import argparse
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fedtrap.config import get_class_names

def plot_convergence(fl_results_path, save_path):
    if not os.path.exists(fl_results_path):
        return
        
    with open(fl_results_path, "r") as f:
        data = json.load(f)
        
    metrics = data.get("metrics_centralized", {})
    acc_data = metrics.get("accuracy", [])
    
    if not acc_data:
        return
        
    rounds = [item[0] for item in acc_data]
    accs = [item[1] for item in acc_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accs, marker='o', linestyle='-', linewidth=2)
    plt.xlabel("Communication Round")
    plt.ylabel("Global Accuracy")
    plt.title("Federated Learning Convergence")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_communication(comm_results_path, raw_images, img_size_kb, save_path):
    if not os.path.exists(comm_results_path):
        return
        
    with open(comm_results_path, "r") as f:
        data = json.load(f)
        
    rounds = [item["round"] for item in data]
    # MB
    bytes_sent = [item["total_bytes_sent"] / (1024 * 1024) for item in data]
    bytes_recv = [item["total_bytes_received"] / (1024 * 1024) for item in data]
    total_bytes = [s + r for s, r in zip(bytes_sent, bytes_recv)]
    
    # Raw transfer estimate (Centralized)
    raw_mb = (raw_images * img_size_kb) / 1024
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, total_bytes, marker='s', label="FL Communication Cost")
    plt.axhline(y=raw_mb, color='r', linestyle='--', label=f"Raw Image Transfer ({raw_mb:.1f} MB)")
    plt.xlabel("Communication Round")
    plt.ylabel("Cumulative Data Transferred (MB)")
    plt.title("Communication Efficiency: Federated vs Centralized")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_per_class_f1(comparison_path, class_names, save_path):
    if not os.path.exists(comparison_path):
        return
        
    df = pd.read_csv(comparison_path)
    
    methods = []
    f1_scores = []
    
    for _, row in df.iterrows():
        method = row["Method"]
        f1_str = row["Per-class F1"]
        if f1_str != "N/A":
            try:
                scores = eval(f1_str)
                methods.append(method)
                f1_scores.append(scores)
            except:
                pass
                
    if not methods:
        return
        
    x = np.arange(len(class_names))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (method, scores) in enumerate(zip(methods, f1_scores)):
        ax.bar(x + i*width - 0.4 + width/2, scores, width, label=method)
        
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 Score by Training Method")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_all(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
    class_names = get_class_names()
    
    plot_convergence(
        os.path.join(results_dir, "federated_results.json"),
        os.path.join(results_dir, "convergence_curve.png")
    )
    
    # Assuming roughly 200 images per client, 50KB each
    plot_communication(
        os.path.join(results_dir, "fl_communication.json"),
        raw_images=400,
        img_size_kb=50,
        save_path=os.path.join(results_dir, "communication_efficiency.png")
    )
    
    plot_per_class_f1(
        os.path.join(results_dir, "comparison.csv"),
        class_names,
        os.path.join(results_dir, "per_class_f1.png")
    )
    
    print(f"Evaluation complete. Plots saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results/')
    args = parser.parse_args()
    
    evaluate_all(args.results_dir)
