<img width="1600" height="870" alt="WhatsApp Image 2026-05-15 at 11 39 09 PM" src="https://github.com/user-attachments/assets/70bdefb8-b353-4fb5-8045-d41671fd42eb" />


https://github.com/user-attachments/assets/3283b217-263a-4b2c-a6c8-f83d919e21de

# Project MORPHOS
**Real-time ambulance detection and automatic traffic signal preemption**

![Status](https://img.shields.io/badge/Status-Working_Prototype-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Arduino](https://img.shields.io/badge/Hardware-Arduino-teal)
![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-yellow)

> Ambulance response time is a measurable factor in patient survival; every minute of delay en route to a cardiac or trauma case increases mortality risk. In dense urban traffic, a meaningful portion of that delay happens at signalized intersections. **Morphos** is a working prototype that uses computer vision to detect approaching ambulances, verify their emergency lights are actually flashing, and preempt the local traffic signal to green — automatically, end-to-end, in about a second.

---

## Demo

<!--
Replace the line below with your uploaded video.
GitHub accepts MP4 uploads directly in markdown — drag-and-drop into the README editor.
Alternative: link to a YouTube/Drive video using the image-thumbnail syntax:
[![Morphos Demo](docs/thumbnail.png)](https://your-video-link)
-->

*[Demo video to be embedded — short clip showing ambulance entering frame, detection bounding box appearing, signal switching to green, and the system returning to normal cycling after the vehicle clears.]*

| Detection in action | Hardware setup |
|---|---|
| ![Detection running on a test clip](docs/detection.png) | ![Arduino + LED traffic-light rig](docs/hardware.png) |

---

## How It Works

**Stage 1 — Object Detection**
YOLOv8-nano detects ambulances in the live camera feed at 30 FPS. In-distribution validation metrics: 99.9% precision, 100% recall, 0.995 mAP@50. See *Important note on metrics* below.

**Stage 2 — Flash Verification**
Detection alone is not enough — a parked ambulance is not an emergency. Two independent checks confirm the vehicle's emergency lights are actively flashing:

- **FFT analysis** — measures brightness oscillation inside the bounding box. Emergency lights in our training footage oscillate at roughly 2–4 Hz; the system accepts a wider 1–6 Hz window to allow for variation between vehicle types and capture rates.
- **HSV color tracking** — counts alternating red and blue pixels over time, providing a second independent signal that the lights are cycling, not steady.

A motion-stability gate suppresses false triggers from fast-moving non-emergency vehicles (bounding-box shift > 40 px between frames).

**Stage 3 — Signal Preemption**
After 25 consecutive confirmed flash frames, a serial command tells the Arduino to switch the light to **GREEN**. When the ambulance clears the frame for 30 frames, normal signal cycling resumes.

A 5-second hardware watchdog timer on the Arduino reverts the light to its default state automatically if the Python process crashes or the serial connection drops.

---

## End-to-End Latency

| Stage | Approximate Latency |
|---|---|
| YOLOv8-nano inference (GPU) | ~15–25 ms per frame |
| Flash-verification window | 833 ms (25 frames @ 30 FPS) |
| Serial transmission (9600 baud) | < 10 ms |
| **Total: ambulance enters frame → light green** | **~0.9–1.0 s** |

Bench-measured on an NVIDIA RTX 3060 Laptop GPU. Real-intersection latency has not been validated.

---

## Architecture

```
Camera (30 FPS) → YOLOv8-Nano (CUDA) → Flash Detector (FFT + HSV) → Serial (9600 baud) → Arduino
                                                                                          ├── RED    → Pin 3
                                                                                          ├── YELLOW → Pin 2
                                                                                          └── GREEN  → Pin 5
```

---

## Training Results

| Metric | Value |
|---|---|
| Model | YOLOv8-nano |
| Training images | 589 |
| Train / Val split | 80 / 20 |
| Epochs | 90 |
| Precision | 0.999 |
| Recall | 1.000 |
| mAP@50 | 0.995 |
| mAP@50–95 | 0.756 |
| Hardware | NVIDIA RTX 3060 Laptop (6 GB) |
| Training time | ~7 minutes |

> **Important note on metrics.** The validation split is drawn from the same recording conditions as the training set (similar weather, time of day, camera angle, and vehicle types). These numbers reflect *in-distribution* performance only. Generalization to unseen conditions — night, rain, different camera placements, different ambulance liveries — has not been validated. Building an out-of-distribution test set is the next planned phase of work.

---

## Limitations & Future Work

Morphos is a working prototype, not a deployed product. Known limitations:

- **Single-condition dataset.** All training footage was captured under daylight, one camera angle, and a narrow weather range. Performance under night, rain, fog, and direct sun glare is unknown.
- **No on-site latency measurement.** End-to-end timing has been measured on bench only. Real-intersection conditions (longer cable runs, controller integration, vehicle approach speeds, larger field of view) have not been tested.
- **Untested failure modes:** vehicle occlusion, multiple emergency vehicles in frame simultaneously, decorative or non-emergency flashing lights, sunset/sunrise reflections that could mimic flash frequencies, and red commercial trucks that resemble ambulance silhouettes.
- **Camera-feed failure not handled distinctly.** The Arduino watchdog handles Python crashes; it does not currently differentiate between a process crash and a camera disconnect mid-emergency.
- **Hardware integration is signal-mimicking, not signal-preempting.** The Arduino drives demonstration LEDs, not an actual traffic-controller cabinet. Production integration with real signal infrastructure would require a controller-specific preemption protocol (e.g., NTCIP, OPTICOM, or a local KSA equivalent).

Planned next steps:

1. Build a held-out test set from independent footage (different camera, location, time of day) and re-report metrics on it.
2. Stress-test against false-positive sources: red trucks, decorative lights, taillight glare, sunset reflections.
3. Document a real-controller integration path for Saudi traffic infrastructure.
4. Measure end-to-end latency on-site at a controlled intersection.

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8-nano (Ultralytics) |
| Computer Vision | OpenCV 4.8+ |
| Signal Processing | NumPy |
| Deep Learning | PyTorch 2.0+ (CUDA) |
| Serial Communication | PySerial (with Heartbeat Keep-alive) |
| Microcontroller | Arduino UNO/Nano |
| Language | Python 3.10, C++ (Arduino) |
| Development assistance | Claude, Roo Code |

---

## Project Structure

| File | Description |
|---|---|
| `morphos_final_v2.py` | Main entry point |
| `config.py` | All tunable parameters |
| `flash_detector.py` | FFT + HSV flash analysis |
| `capture_dataset.py` | Training-data capture tool |
| `label_tool.py` | Bounding-box labeling GUI |
| `train.py` | YOLOv8 training pipeline |
| `morphos_serial_test.py` | Arduino serial tester |
| `test_final.py` | Inference demo (camera or single image) |
| `models/trained/best.pt` | Trained model weights |
| `morphos_traffic_controller/` | Arduino firmware (.ino) |

---

## Quick Start

**Requirements:** Python 3.10+, USB webcam, Arduino UNO/Nano. NVIDIA GPU strongly recommended for real-time inference; CPU mode runs at lower FPS and is suitable for offline testing only.

```bash
git clone https://github.com/Lashiien/morphos.git
cd morphos
pip install -r requirements.txt
```

Upload `morphos_traffic_controller/morphos_traffic_controller.ino` to your Arduino via the Arduino IDE, then run:

```bash
python morphos_final_v2.py
```

| Key | Action |
|---|---|
| `Q` | Quit |
| `C` | Force clear emergency mode |

> If no Arduino is connected, the system runs in debug mode — detection and flash verification still work, but hardware signal control is skipped.

---

## Retraining (Optional)

```bash
# 1. Capture training images (SPACE to capture, Q to quit)
python capture_dataset.py

# 2. Label images (draw a box with the mouse, N = next, Q = quit)
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
| `FLASH_FREQUENCY_RANGE` | `(1.0, 6.0)` | Emergency-light frequency window (Hz) |
| `FLASH_THRESHOLD_STD` | `15.0` | Flash brightness sensitivity |
| `FLASH_CONFIRMATION_FRAMES` | `25` | Frames to confirm emergency (~0.8 s @ 30 FPS) |
| `FRAMES_TO_CLEAR` | `30` | Frames without flash before exiting emergency mode |
| `SERIAL_BAUD_RATE` | `9600` | Arduino communication speed |

---

## Author

Built by **Ahmed Lashin** — Industrial Engineering undergraduate
Contact: [aali932003@gmail.com](mailto:aali932003@gmail.com) · [LinkedIn](https://www.linkedin.com/in/lashien/)

---

## License

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Commercial use requires written permission from the author.
