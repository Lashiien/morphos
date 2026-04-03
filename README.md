# 🚑 Project MORPHOS

**Real-Time Ambulance Detection & Emergency Traffic Signal Control**

An embedded AI system that detects active emergency vehicles using computer vision and automatically controls traffic signals to create priority passage at intersections.

![Status](https://img.shields.io/badge/Status-Working_Prototype-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Arduino](https://img.shields.io/badge/Hardware-Arduino-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📹 Demo

> **Demo video coming soon** — will show real-time ambulance detection triggering physical traffic light changes via Arduino.

<!-- TODO: Replace this section with demo GIF or YouTube link -->

---

## 🧠 How It Works

The system uses a **three-stage detection pipeline**:
USB Camera → YOLOv8 Detection → Flash Analysis → Arduino → Traffic Lights


### Stage 1 — Object Detection (YOLOv8)
A YOLOv8-nano model trained on 589 labeled images detects ambulances in the video feed with **99.9% precision** and **100% recall** (mAP50: 0.995).

### Stage 2 — Emergency Flash Verification
Once an ambulance is detected, the system analyzes the bounding box region using two methods:

- **FFT Frequency Analysis**: Measures brightness oscillation within the ROI. Emergency lights flash at 1–6 Hz — the system uses Fast Fourier Transform to identify this signature pattern.
- **HSV Color Alternation**: Tracks red (H: 0-10, 170-180) and blue (H: 100-130) pixel counts over time. Alternating red/blue patterns provide secondary confirmation of active emergency lights.

A **motion stability gate** prevents false triggers from fast-moving objects — flash analysis is skipped when the bounding box moves more than 40 pixels between frames.

### Stage 3 — Traffic Signal Control
When emergency mode is confirmed (25 consecutive frames of verified flashing ≈ 0.8 seconds), the system sends a serial command to an Arduino, which immediately switches the traffic light to **GREEN** for the emergency vehicle. When the ambulance leaves or lights stop flashing, normal signal cycling resumes.

A **hardware watchdog timer** (5 seconds) on the Arduino ensures the traffic light returns to normal operation even if the Python application crashes.

---

## 🏗️ Architecture
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  USB Camera  │────▶│  YOLOv8 Nano     │────▶│  Flash Detector    │
│  (30 FPS)    │     │  (CUDA/RTX 3060) │     │  FFT + HSV + Motion│
└─────────────┘     └──────────────────┘     └────────┬───────────┘
│
┌────────▼───────────┐
│  Serial Controller │
│  (9600 baud)       │
└────────┬───────────┘
│
┌────────▼───────────┐
│  const int RED_PIN = 3;
const int YELLOW_PIN = 2;
const int GREEN_PIN = 5;   │
└────────────────────┘


---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Model | YOLOv8-nano |
| Training Images | 589 (single class) |
| Train/Val Split | 80/20 |
| Epochs | 90 |
| Precision | 0.999 |
| Recall | 1.000 |
| mAP50 | 0.995 |
| mAP50-95 | 0.756 |
| GPU | NVIDIA RTX 3060 Laptop (6GB) |
| Training Time | ~7 minutes |

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8-nano (Ultralytics) |
| Computer Vision | OpenCV 4.8+ |
| Signal Processing | SciPy (FFT), NumPy |
| Deep Learning | PyTorch 2.0+ (CUDA) |
| Serial Communication | PySerial |
| Microcontroller | Arduino UNO/Nano |
| Language | Python 3.10, C++ (Arduino) |

---

## 📁 Project Structure

morphos/
├── morphos_final_v2.py          # Main application (entry point)
├── config.py                     # All configurable parameters
├── flash_detector.py             # FFT + HSV flash analysis engine
├── capture_dataset.py            # Training data capture tool
├── label_tool.py                 # Bounding box labeling GUI
├── train.py                      # YOLOv8 training pipeline
├── requirements.txt              # Python dependencies
├── models/
│   └── trained/
│       └── best.pt               # Trained YOLOv8 weights
└── morphos_traffic_controller/
└── morphos_traffic_controller.ino  # Arduino firmware


---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- Arduino UNO/Nano with traffic light circuit
- USB webcam

### Installation

```bash
git clone https://github.com/Lashien/morphos.git
cd morphos
pip install -r requirements.txt

Upload Arduino Firmware
Open morphos_traffic_controller/morphos_traffic_controller.ino in Arduino IDE
Connect Arduino via USB
Upload the sketch

Run
<BASH>
python morphos_final_v2.py
Key	Action
Q	Quit
C	Force clear emergency mode

The system auto-detects the Arduino serial port. If no Arduino is connected, it runs in debug mode (detection works, just no hardware control).

🔄 Retrain the Model (Optional)
If you want to train on your own ambulance/vehicle:

<BASH>
# 1. Capture training images
python capture_dataset.py
# Press SPACEBAR to capture, Q to quit

# 2. Label images
python label_tool.py
# Draw box with mouse, N = next image, Q = quit

# 3. Train
python train.py
# Weights auto-save to models/trained/best.pt
⚙️ Configuration

All tunable parameters are in config.py:

Parameter	Default	Description
CONFIDENCE_THRESHOLD	0.5	Minimum YOLO detection confidence
FLASH_FREQUENCY_RANGE	(1.0, 6.0)	Emergency light frequency range (Hz)
FLASH_THRESHOLD_STD	15.0	Flash brightness sensitivity
FLASH_CONFIRMATION_FRAMES	25	Frames needed to confirm emergency (~0.8s)
FRAMES_TO_CLEAR_EMERGENCY	30	Frames without flash to exit emergency mode
SERIAL_BAUD_RATE	9600	Arduino communication speed
🤝 Contributors
Lashien — Developer
More contributors to be added
🤖 Built With AI
This project was developed with the assistance of AI tools (Claude, Roo Code) for code architecture, debugging, and documentation. All core engineering decisions, hardware assembly, data collection, and testing were performed by the developer.

📄 License
MIT License — see LICENSE for details.
