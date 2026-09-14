# Federated Robotic Smart Traps for Selective Pest Control (FedTrap)

**B.Tech CSE (AI & Robotics) Final Year Project**

FedTrap is an intelligent, edge-AI powered robotic trap system designed for selective agricultural pest control. Unlike traditional indiscriminate traps, FedTrap uses computer vision (MobileNetV2) running locally on a Raspberry Pi to distinguish between agricultural pests (which it captures) and beneficial insects (which it releases). It also incorporates a simulated Federated Learning architecture to allow multiple distributed traps to securely share learning weights without transferring raw images.

---

## 🧠 System Architecture

The software is designed to be lightweight, modular, and optimized for Edge AI inference on the Raspberry Pi 4 (8GB).

### Hardware Stack
* **Compute**: Raspberry Pi 4 Model B (8 GB)
* **Vision**: Raspberry Pi Camera Module 3
* **Actuation**: SG90 Micro Servo Motor (GPIO 18)

### Software Stack
* **AI Model**: MobileNetV2 (Pre-trained on ImageNet, fine-tuned for 5 insect classes)
* **Computer Vision**: OpenCV (Motion detection & bounding boxes)
* **Control Logic**: Temporal Confirmation Filter (requires 3 consecutive confident frames before physical actuation to prevent false positives)

---

## 📂 Project Directory Structure & Important Files

All critical parameters (pins, thresholds, insect classes) are centralized in `config/config.yaml`. **Do not hardcode values in the Python scripts!**

```text
project files/
│
├── config/
│   └── config.yaml             # ⚙️ THE BRAIN: Edit classes, servo angles, GPIO pins, and thresholds here.
│
├── fedtrap/                    # 📦 CORE MODULES (The engine of the project)
│   ├── camera.py               # Handles Video streams (Webcam & PiCamera 3) + Motion Detection
│   ├── servo_controller.py     # GPIO PWM logic to move the SG90 Servo (Capture/Release)
│   ├── inference.py            # Loads the .pt model and runs image classification
│   ├── temporal_filter.py      # State machine: Ensures 3 consecutive frames before acting
│   └── decision_policy.py      # Maps the classified insect to an ACTION (e.g., Moth -> CAPTURE)
│
├── scripts/                    # 🚀 EXECUTABLE SCRIPTS (Run these from terminal)
│   ├── camera_inference.py     # 🎥 LIVE DEMO: Run this to test the live webcam pipeline
│   ├── test_folder.py          # Batch testing script to evaluate a folder of images
│   └── benchmark_pi.py         # Measures FPS and latency on the Raspberry Pi
│
├── notebooks/                  # ☁️ CLOUD TRAINING CODE
│   ├── kaggle_cell1_dataset.py # Script used to download & assemble 5,000 images on Kaggle
│   └── kaggle_cell2_training.py# Script used to train the model and simulate Federated Learning
│
├── models/                     # 🧠 TRAINED WEIGHTS
│   └── classifier_best.pt      # The trained MobileNetV2 weights (96.4% Kaggle accuracy)
│
└── results/                    # 📊 OUTPUTS
    └── inference_log.csv       # Logs every insect detected and captured during live runs
```

---

## 🐞 Insect Classifications & Actions

The model is trained to recognize 5 specific insect classes, mapping them to actionable decisions.

| Class | Category | Trap Action | Servo Angle |
| :--- | :--- | :--- | :--- |
| **Beetle** | PEST | `CAPTURE` | 150° |
| **Moth** | PEST | `CAPTURE` | 150° |
| **Grasshopper** | PEST | `CAPTURE` | 150° |
| **Butterfly** | BENEFICIAL | `RELEASE` | 30° |
| **Honeybee** | BENEFICIAL | `RELEASE` | 30° |

*(Default idle/home position for the servo is 90°)*

---

## 🚀 How to Run the Project (Commands)

### 1. Live PC Demo (Webcam)
To test the full visual pipeline and decision logic on your laptop before deploying to the Pi:
```bash
python scripts/camera_inference.py --source webcam --model models/classifier_best.pt --display
```
*Hold up pictures of insects to your webcam. The terminal will log `ACTUATE: [class] -> [action]`.*

### 2. Batch Evaluate a Test Folder
If you have a folder named `Testing` containing subfolders of insect images (e.g., `Testing/beetle`, `Testing/honeybee`), run:
```bash
python scripts/test_folder.py --test-dir Testing --model models/classifier_best.pt
```

### 3. Raspberry Pi Live Deployment
Once the code is moved to the Raspberry Pi, run the same inference script, but change the source to `picamera`:
```bash
python scripts/camera_inference.py --source picamera --model models/classifier_best.pt
```

---

## 📈 Presentation & Demo Tips (The "Cheese")
Because the model was trained on specific visual distributions (e.g., top-down honeybee hives, moths resting on bark), it may struggle with random Google images containing pure white backgrounds or macro extreme close-ups. 

**For a flawless live presentation:**
Use images that visually match the training datasets. Search for images like `"honeybee on honeycomb"`, `"luna moth flat on bark"`, or `"scarab beetle on dirt"`. This guarantees high-confidence actuations for the live hardware demo.
