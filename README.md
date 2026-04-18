# Project MORPHOS
**Real-time ambulance detection + automatic traffic signal control**

![Status](https://img.shields.io/badge/Status-Working_Prototype-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Arduino](https://img.shields.io/badge/Hardware-Arduino-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)

A camera detects an active ambulance, verifies its emergency lights are flashing, and sends a signal to an Arduino to turn the traffic light green — all in under a second.

---

## How It Works

**Stage 1 — Object Detection**
YOLOv8-nano detects ambulances in the live camera feed (99.9% precision, 100% recall).

**Stage 2 — Flash Verification**
Two checks confirm the ambulance's emergency lights are actually active:
- **FFT analysis** — measures brightness oscillation in the bounding box region. Emergency lights flash at 1–6 Hz.
- **HSV color tracking** — detects alternating red/blue pixel counts over time.

A motion stability gate ignores fast-moving objects (bounding box shift > 40px between frames) to avoid false triggers.

**Stage 3 — Signal Control**
After 25 consecutive confirmed flash frames (~0.8s), a serial command tells the Arduino to switch the light to **GREEN**. When the ambulance clears, normal cycling resumes.

A hardware watchdog timer (5s) on the Arduino resets the light automatically if the Python app crashes.

---

## Architecture

```
Camera (30fps) → YOLOv8-Nano (CUDA) → Flash Detector (FFT + HSV) → Serial (9600 baud) → Arduino
                                                                                          ├── RED    → Pin 3
                                                                                          ├── YELLOW → Pin 2
                                                                                          └── GREEN  → Pin 5
```

---

## Training Results

| Metric | Value |
|---|---|
| Model | YOLOv8-nano |
| Training Images | 589 |
| Train/Val Split | 80/20 |
| Epochs | 90 |
| Precision | 0.999 |
| Recall | 1.000 |
| mAP50 | 0.995 |
| mAP50-95 | 0.756 |
| GPU | NVIDIA RTX 3060 Laptop (6GB) |
| Training Time | ~7 minutes |

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8-nano (Ultralytics) |
| Computer Vision | OpenCV 4.8+ |
| Signal Processing | SciPy (FFT), NumPy |
| Deep Learning | PyTorch 2.0+ (CUDA) |
| Serial Communication | PySerial |
| Microcontroller | Arduino UNO/Nano |
| Language | Python 3.10, C++ (Arduino) |

---

## Project Structure

| File | Description |
|---|---|
| `morphos_final_v2.py` | Main entry point |
| `config.py` | All tunable parameters |
| `flash_detector.py` | FFT + HSV flash analysis |
| `capture_dataset.py` | Training data capture tool |
| `label_tool.py` | Bounding box labeling GUI |
| `train.py` | YOLOv8 training pipeline |
| `morphos_serial_test.py` | Arduino serial tester |
| `test_final.py` | Simple inference demo |
| `models/trained/best.pt` | Trained model weights |
| `morphos_traffic_controller/` | Arduino firmware (.ino) |

---

## Quick Start

**Requirements:** Python 3.10+, NVIDIA GPU (recommended), Arduino UNO/Nano, USB webcam

```bash
git clone https://github.com/Lashiien/morphos.git
cd morphos
pip install -r requirements.txt
```

Upload `morphos_traffic_controller/morphos_traffic_controller.ino` to your Arduino via Arduino IDE, then:

```bash
python morphos_final_v2.py
```

| Key | Action |
|---|---|
| `Q` | Quit |
| `C` | Force clear emergency mode |

> If no Arduino is connected, the system runs in debug mode — detection still works, no hardware control.

---

## Retraining (Optional)

```bash
# 1. Capture training images (SPACE to capture, Q to quit)
python capture_dataset.py

# 2. Label images (draw box with mouse, N = next, Q = quit)
python label_tool.py

# 3. Train (weights save to models/trained/best.pt)
python train.py
```

---

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum YOLO detection confidence |
| `FLASH_FREQUENCY_RANGE` | `(1.0, 6.0)` | Emergency light frequency range (Hz) |
| `FLASH_THRESHOLD_STD` | `15.0` | Flash brightness sensitivity |
| `FLASH_CONFIRMATION_FRAMES` | `25` | Frames to confirm emergency (~0.8s) |
| `FRAMES_TO_CLEAR_EMERGENCY` | `30` | Frames without flash to exit emergency mode |
| `SERIAL_BAUD_RATE` | `9600` | Arduino communication speed |

---

## Credits

Built by **Lashien**.
Developed with AI assistance (Claude, Roo Code) for architecture, debugging, and documentation. All engineering decisions, hardware assembly, data collection, and testing by the developer.

---

## License

>  Licensed under CC BY-NC 4.0 | 
Commercial use requires permission
