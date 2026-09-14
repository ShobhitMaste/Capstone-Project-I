import os
import sys
import time
import argparse
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedtrap.inference import InsectClassifier
from fedtrap.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Performance Benchmarking")
    parser.add_argument("--model", type=str, default="models/classifier_best.pt", help="Path to model")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Detect Platform
    is_pi = False
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'Raspberry Pi' in f.read() or 'BCM' in f.read() or 'aarch64' in os.uname().machine:
                is_pi = True
    except:
        pass
        
    platform = "PI" if is_pi else "DESKTOP"
    print(f"Detected Platform: {platform}")
    print(f"Target Device: {device}")
    
    classifier = InsectClassifier(model_path=args.model, device=str(device))
    
    dummy_img = Image.new('RGB', (args.img_size, args.img_size))
    
    print("Warming up...")
    for _ in range(10):
        classifier.predict(dummy_img)
        
    print(f"Running {args.num_iterations} iterations...")
    
    prep_times = []
    inf_times = []
    post_times = []
    
    # Check memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    gpu_mem_before = torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
    
    for i in range(args.num_iterations):
        t0 = time.time()
        tensor = classifier.preprocess(dummy_img)
        t1 = time.time()
        
        if classifier.is_onnx:
            ort_inputs = {classifier.ort_session.get_inputs()[0].name: tensor.numpy()}
            ort_outs = classifier.ort_session.run(None, ort_inputs)
            output = torch.tensor(ort_outs[0])
        else:
            tensor = tensor.to(classifier.device)
            with torch.no_grad():
                output = classifier.model(tensor)
        t2 = time.time()
        
        probs = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        pred_idx = np.argmax(probs)
        conf = float(probs[pred_idx])
        t3 = time.time()
        
        prep_times.append((t1 - t0) * 1000)
        inf_times.append((t2 - t1) * 1000)
        post_times.append((t3 - t2) * 1000)
        
    mem_after = process.memory_info().rss / (1024 * 1024)
    gpu_mem_after = torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
    
    avg_prep = np.mean(prep_times)
    avg_inf = np.mean(inf_times)
    avg_post = np.mean(post_times)
    
    std_prep = np.std(prep_times)
    std_inf = np.std(inf_times)
    std_post = np.std(post_times)
    
    avg_total = avg_prep + avg_inf + avg_post
    fps = 1000.0 / avg_total if avg_total > 0 else 0
    
    results = {
        "platform": platform,
        "device": str(device),
        "iterations": args.num_iterations,
        "preprocessing_ms": {"mean": avg_prep, "std": std_prep},
        "inference_ms": {"mean": avg_inf, "std": std_inf},
        "postprocessing_ms": {"mean": avg_post, "std": std_post},
        "total_ms": {"mean": avg_total},
        "fps": fps,
        "memory_mb": mem_after,
        "gpu_memory_mb": gpu_mem_after if torch.cuda.is_available() else None
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Benchmark Results ---")
    print(f"{'Component':<18} | {'Avg Time (ms)':<15} | {'Std Dev (ms)':<15}")
    print("-" * 55)
    print(f"{'Preprocessing':<18} | {avg_prep:<15.2f} | {std_prep:<15.2f}")
    print(f"{'Inference':<18} | {avg_inf:<15.2f} | {std_inf:<15.2f}")
    print(f"{'Postprocessing':<18} | {avg_post:<15.2f} | {std_post:<15.2f}")
    print("-" * 55)
    print(f"{'Total Time':<18} | {avg_total:<15.2f} | {'-':<15}")
    print(f"{'FPS':<18} | {fps:<15.2f} | {'-':<15}")
    print(f"{'RAM Usage (MB)':<18} | {mem_after:<15.2f} | {'-':<15}")
    if torch.cuda.is_available():
        print(f"{'GPU Mem (MB)':<18} | {gpu_mem_after:<15.2f} | {'-':<15}")

if __name__ == "__main__":
    main()
