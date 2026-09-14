# FedTrap — Federated Robotic Smart Traps for Selective Pest Control

> **B.Tech CSE (AI & Robotics) Final-Year Project**
> 3-person team · 2 physical Raspberry Pi trap nodes · Federated Learning

---

## 1. Project Overview

FedTrap is a federated edge-AI system for selective insect pest control. Two physical robotic traps, each powered by a Raspberry Pi 4 (8 GB), use on-device image classification to distinguish **pest** insects from **beneficial** ones and actuate a servo-driven gate accordingly — capturing pests while safely releasing pollinators.

The traps collaborate via **Flower-based Federated Learning (FedAvg)**, exchanging only model parameter updates over local Wi-Fi. **Raw images never leave the device.** This architecture enables collaborative model improvement across distributed, non-IID observation sites without centralizing sensitive visual data.

### Key Research Question

> *Does federated collaboration improve cross-node generalization when each trap sees a different (non-IID) distribution of insects?*

### Five-Class Prototype

| Class       | Category   | Action  |
|-------------|------------|---------|
| Grasshopper | PEST       | CAPTURE |
| Beetle      | PEST       | CAPTURE |
| Moth        | PEST       | CAPTURE |
| Honeybee    | BENEFICIAL | RELEASE |
| Butterfly   | BENEFICIAL | RELEASE |

Unknown / low-confidence predictions → **NO ACTION** (safe default).

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LOCAL WI-FI                            │
│                                                             │
│   ┌──────────────────────┐    ┌──────────────────────┐     │
│   │    TRAP A (Pi 4)     │    │    TRAP B (Pi 4)     │     │
│   │                      │    │                      │     │
│   │  Pi Camera Module 3  │    │  Pi Camera Module 3  │     │
│   │         ↓            │    │         ↓            │     │
│   │  Motion Detection    │    │  Motion Detection    │     │
│   │         ↓            │    │         ↓            │     │
│   │  MobileNetV2 (224px) │    │  MobileNetV2 (224px) │     │
│   │         ↓            │    │         ↓            │     │
│   │  Temporal Filter     │    │  Temporal Filter     │     │
│   │         ↓            │    │         ↓            │     │
│   │  Decision Policy     │    │  Decision Policy     │     │
│   │         ↓            │    │         ↓            │     │
│   │  SG90 Servo Gate     │    │  SG90 Servo Gate     │     │
│   │                      │    │                      │     │
│   │  ┌────────────────┐  │    │  ┌────────────────┐  │     │
│   │  │ Flower Client A │◄─┼────┼─►│ Flower Client B │  │     │
│   │  └────────┬───────┘  │    │  └────────────────┘  │     │
│   │           │          │    │                      │     │
│   │  ┌────────▼───────┐  │    │                      │     │
│   │  │ Flower Server  │  │    │                      │     │
│   │  │  (FedAvg)      │  │    │                      │     │
│   │  └────────────────┘  │    │                      │     │
│   └──────────────────────┘    └──────────────────────┘     │
│                                                             │
│   Model parameters only (~5 MB/round) ◄──►                 │
│   NO raw images exchanged                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Hardware

| Component            | Spec                         | Qty |
|----------------------|------------------------------|-----|
| Raspberry Pi 4       | 8 GB RAM, 64-bit OS          | 2   |
| Pi Camera Module 3   | CSI interface                | 2   |
| SG90 Micro Servo     | 1-DoF gate mechanism         | 2   |
| Power Supply         | 5V regulated (servo separate)| 2   |
| Trap Enclosure       | Cardboard/Tupperware chamber | 2   |

> ⚠️ **Important**: Do NOT power the SG90 directly from the Pi 5V pin. Use an external regulated 5V supply with common ground to prevent brownouts.

---

## 4. Dataset

Sources (see [DATASETS.md](DATASETS.md) for full details):

| Source                         | Classes Provided         |
|--------------------------------|--------------------------|
| Insects Recognition (Kaggle)   | Grasshopper, Butterfly   |
| ArTaxOr (Kaggle)               | Beetle (bbox-cropped)    |
| BeeImage (Kaggle)              | Honeybee                 |
| Butterfly & Moths 100 species  | Moth, Butterfly          |
| IP102 (optional)               | Top-up for pest classes  |
| Trap-camera images             | All classes (highest value) |

Target: ≥300 images/class for initial training.

```
data/
├── dataset/
│   ├── grasshopper/
│   ├── beetle/
│   ├── moth/
│   ├── honeybee/
│   └── butterfly/
└── trap_val/          # actual trap-camera images (separate)
```

---

## 5. Installation

### Desktop / Laptop (Training + Development)

```bash
# Clone or navigate to project
cd "project files"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Raspberry Pi

```bash
# Install PyTorch ARM wheel first (see pytorch.org)
pip install -r requirements-pi.txt
```

---

## 6. Dataset Preparation

```bash
# Step 1: Download datasets (requires Kaggle API token)
python scripts/download_datasets.py

# Step 2: Validate + split into train/val/test
python scripts/prepare_dataset.py
```

This generates:
- `data/split_metadata.json` — deterministic 70/15/15 split
- `data/dataset_report.txt` — per-class counts
- `data/class_counts.json` — machine-readable counts

---

## 7. Training

```bash
python scripts/train_classifier.py \
    --data data/split_metadata.json \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.001 \
    --img-size 224 \
    --seed 42
```

Two-phase transfer learning:
1. **Phase 1** (5 epochs): Backbone frozen → train classifier head
2. **Phase 2** (remaining): Unfreeze top layers → fine-tune

Outputs:
- `models/classifier_best.pt` — best validation model
- `models/classifier.pt` — final model
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/classification_report.txt`
- `results/training_metrics.json`

---

## 8. Evaluation

```bash
python scripts/evaluate.py \
    --model models/classifier_best.pt \
    --data data/split_metadata.json \
    --split test
```

---

## 9. Camera Inference

### Desktop (video file)
```bash
python scripts/camera_inference.py --source video.mp4 --display
```

### Desktop (webcam)
```bash
python scripts/camera_inference.py --source webcam --display
```

### Raspberry Pi
```bash
python scripts/camera_inference.py --source picamera
```

---

## 10. Single-Image Prediction

```bash
python scripts/predict.py --image path/to/insect.jpg
```

Output:
```
Predicted: grasshopper
Confidence: 0.94

All probabilities:
  grasshopper: 0.94
  beetle:      0.02
  moth:        0.01
  honeybee:    0.02
  butterfly:   0.01

Category: PEST
Action: CAPTURE
```

---

## 11. Raspberry Pi Setup

1. Flash Raspberry Pi OS 64-bit
2. Install Python 3.9+ and pip
3. Install `requirements-pi.txt`
4. Copy trained model to `models/`
5. Connect Pi Camera Module 3 (CSI)
6. Connect SG90 servo to GPIO 18 (configurable in `config/config.yaml`)
7. Set `trap.id` and `trap.role` in config for each Pi

---

## 12. Servo Setup

GPIO wiring (SG90):
- **Signal** → GPIO 18 (configurable)
- **VCC** → External 5V supply (NOT Pi 5V)
- **GND** → Common ground with Pi

Configure in `config/config.yaml`:
```yaml
servo:
    gpio: 18
    home_angle: 90
    capture_angle: 150
    release_angle: 30
```

Calibrate angles empirically based on your trap's mechanical geometry.

---

## 13. Federated Learning Setup

### Simulation (single machine)
```bash
python experiments/run_federated.py --rounds 10 --skew 0.8
```

### Two-Pi Real Federation

**On Trap A (server + client):**
```bash
python federated/server.py --rounds 10
```

**On Trap B (client only):**
```bash
# Set federated.server_address to Trap A's IP in config
python federated/client.py
```

---

## 14. Non-IID Experiment

```bash
# Run all baselines: LOCAL vs CENTRALIZED vs FEDERATED
python experiments/run_comparison.py --rounds 10 --skew 0.8 --seed 42
```

Generates:
- `results/local_results.json`
- `results/centralized_results.json`
- `results/federated_results.json`
- `results/comparison.csv`
- `results/comparison.png`

### Comparison Table Format

| Method       | Accuracy | Macro-F1 | Communication | Rounds | Client A F1 | Client B F1 |
|--------------|----------|----------|---------------|--------|-------------|-------------|
| Local-Only   | —        | —        | 0 B           | —      | —           | —           |
| Centralized  | —        | —        | N/A           | —      | —           | —           |
| FedAvg       | —        | —        | ~50 MB        | 10     | —           | —           |
| FedProx      | —        | —        | ~50 MB        | 10     | —           | —           |

---

## 15. Dashboard

```bash
python dashboard/app.py --port 5000
```

Open `http://localhost:5000` for real-time monitoring.

> The dashboard is optional — the robot operates without it.

---

## 16. Testing

```bash
python -m pytest tests/ -v
```

Tests cover:
- Decision policy (class→action, thresholds, unknown handling)
- Temporal filter (state machine, multi-frame confirmation)
- Servo controller (mock state transitions)
- Model (output shape, freeze/unfreeze, save/load)
- Config (YAML parsing, typed accessors)
- FedAvg (weighted averaging correctness)
- Non-IID partitioning (reproducibility, properties)

---

## 17. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: fedtrap` | Run from project root, or add to PYTHONPATH |
| Camera not found on Pi | Enable camera in `raspi-config`, reboot |
| Servo jittering | Use `pigpio` instead of `RPi.GPIO`; ensure separate power |
| CUDA out of memory | Reduce `--batch-size` or use CPU |
| Kaggle download fails | Check `~/.kaggle/access_token` |
| Low accuracy | Ensure ≥300 images/class; check for corrupt images |

---

## 18. Limitations

This project does **NOT** claim:
- ❌ First AI insect trap
- ❌ First selective insect trap
- ❌ That FL itself is novel
- ❌ That federated learning guarantees privacy
- ❌ That five classes represent all agricultural insects
- ❌ That benchmark accuracy equals field performance

This project **does** demonstrate:
- ✅ Integration of physical robotic trap nodes with on-device AI
- ✅ Federated collaborative learning without raw image exchange
- ✅ Evaluation under controlled non-IID insect distributions
- ✅ Real-time selective physical actuation based on classification
- ✅ Experimentally evaluated LOCAL vs CENTRALIZED vs FL comparison

---

## 19. Research Methodology

1. **Dataset**: Curated from verified public sources + trap-camera images
2. **Split**: Deterministic 70/15/15 train/val/test before any augmentation
3. **Non-IID**: Dirichlet-controlled label-skew partitioning (configurable α)
4. **Baselines**: Local-only, Centralized, FedAvg, (optional FedProx)
5. **Metrics**: Accuracy, per-class P/R/F1, macro-F1, confusion matrix, communication cost
6. **Seeds**: All experiments use deterministic seeds for reproducibility
7. **Hardware**: Results labeled as DESKTOP or PI — never fabricated

---

## 20. Ethical Considerations

- Asymmetric risk: killing a beneficial insect (pollinator) is ecologically worse than missing a pest
- Unknown/low-confidence → safe default (NO ACTION / RELEASE)
- FL communication: "raw training images are not exchanged" (not "privacy guaranteed")
- No differential privacy or secure aggregation implemented (out of scope for Project-I)

---

## Project Structure

```
project files/
├── config/config.yaml          # Central configuration
├── data/dataset/               # Training images (5 class folders)
├── data/trap_val/              # Trap-camera images (separate)
├── models/                     # Saved model weights
├── results/                    # Training curves, metrics, FL results
├── scripts/                    # CLI scripts (train, evaluate, predict, etc.)
├── experiments/                # FL experiments + comparisons
├── fedtrap/                    # Core library (model, policy, servo, camera)
├── federated/                  # Flower FL (client, server, fedavg, fedprox)
├── dashboard/                  # Flask monitoring dashboard
├── tests/                      # Unit tests
├── DATASETS.md                 # Dataset source guide
├── labels.json                 # Class↔index mapping
├── requirements.txt            # Desktop dependencies
├── requirements-pi.txt         # Pi-only dependencies
└── README.md                   # This file
```

---

## License

Academic project — B.Tech CSE (AI & Robotics), Semester 7.
