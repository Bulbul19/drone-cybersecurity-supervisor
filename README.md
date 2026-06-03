# GPS Spoofing-Resistant Drone Navigation System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204B-red)
![Hardware](https://img.shields.io/badge/Hardware-GPS%20%7C%20IMU%20%7C%20Barometer-green)
![Accuracy](https://img.shields.io/badge/Detection%20Accuracy-90.76%25-brightgreen)
![Status](https://img.shields.io/badge/Research%20Paper-In%20Preparation-orange)

A real-time, hardware-deployed GPS spoofing detection system for UAVs using Adaptive Neuro-Fuzzy Inference System (ANFIS) and multi-sensor fusion. Built and tested on a Raspberry Pi 4 Model B with live GPS, IMU, and barometer sensors.

---

## 📌 Problem Statement

GPS spoofing attacks inject false position signals to mislead autonomous drones, causing navigation failures, physical hijacking, or crashes. Most existing UAV navigation systems rely solely on GPS without cross-validating against onboard sensors — making them inherently vulnerable.

This project addresses that gap by implementing a trust-based anomaly detection layer that continuously cross-validates GPS data against IMU and barometer readings. When inconsistencies are detected, the supervisor flags the GPS signal as compromised and can trigger a failsafe response.

---

## 🔧 Hardware Setup

| Component | Model | Interface |
|-----------|-------|-----------|
| Single-board Computer | Raspberry Pi 4 Model B | — |
| GPS Module | Neo-6M | UART (`/dev/ttyAMA0`, 9600 baud) |
| IMU Sensor | MPU-6050 | I2C (Bus 1) |
| Barometer | BMP280 | I2C (Bus 1) |

### Wiring Overview
```
Raspberry Pi 4B
├── UART (GPIO 14/15) ──► Neo-6M GPS Module
└── I2C  (GPIO 2/3)  ──► MPU-6050 (IMU)
                     └── BMP280  (Barometer)
```

---

## 🧠 System Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Neo-6M GPS │    │  MPU-6050   │    │   BMP280    │
│   Module    │    │     IMU     │    │  Barometer  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                 ┌────────▼────────┐
                 │  Sensor Fusion  │
                 │  fuzzy_fusion   │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │   ANFIS Model   │
                 │  (PyTorch)      │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │   Trust Score   │
                 │   (0.0 – 1.0)   │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │   Supervisor    │──► ALERT / SAFE
                 │  live_supervisor│
                 └─────────────────┘
```

---

## ⚙️ Features

- **Real-time GPS monitoring** via Neo-6M over UART
- **IMU-based motion validation** using MPU-6050 accelerometer and gyroscope data
- **Barometric cross-validation** using BMP280 altitude readings
- **ANFIS-based trust scoring** — continuous 0.0–1.0 confidence score per GPS reading
- **Fuzzy logic fusion** — handles sensor uncertainty and noise gracefully
- **Live supervisor decisions** — flags spoofed signals in real time
- **CSV logging** — all sensor readings and trust scores logged for analysis

---

## 📹 Demo

Live demonstration of the system detecting GPS spoofing on Raspberry Pi 4B:

[![GPS Spoofing Detection Demo](https://img.youtube.com/vi/B0zH2406mbA/0.jpg)](https://youtube.com/shorts/B0zH2406mbA)

> Real-time trust score dropping as GPS spoofing is introduced — system flags the attack within seconds.

---

## 📊 Results

### Detection Performance

| Metric | Value |
|--------|-------|
| Detection Accuracy | **90.76%** |
| Model Architecture | ANFIS (PyTorch) |
| Training Epochs | 200 |
| Test Environment | Live hardware on Raspberry Pi 4B |
| Sensors Used | GPS + IMU + Barometer |

### Test Scenarios

| Scenario | Description | Duration (min) |
|----------|-------------|----------------|
| S1 | Normal GPS (baseline) | 45.62 |
| S2 | Gradual Spoofing | 13.50 |
| S3 | Sudden Spoofing | 3.15 |
| S4 | GPS Jamming | 2.33 |

The system successfully detected both gradual and sudden spoofing patterns. Gradual spoofing (S2) is the harder case as signal deviation is slow — the ANFIS model's neuro-fuzzy reasoning layer handles this better than threshold-based approaches.

---

## 📁 Repository Structure

```
drone-cybersecurity-supervisor/
├── src/                        # Core application source code
│   ├── sensors.py              # Sensor interface layer (GPS, IMU, Barometer)
│   ├── imu_reader.py           # MPU-6050 I2C reader
│   ├── gps_realtime_v3.py      # Neo-6M UART GPS reader
│   ├── fuzzy_engine.py         # Fuzzy logic inference engine
│   ├── fuzzy_fusion.py         # Multi-sensor fusion logic
│   ├── live_supervisor.py      # Real-time supervisor and alert system
│   ├── master_supervisor_v3.py # Master controller (entry point)
│   └── launcher.py             # System launcher
│
├── models/                     # Trained ANFIS model files
│   ├── anfis_v3.pth            # Primary trained model (PyTorch)
│   ├── anfis_v3_meta.json      # Model metadata and normalization params
│   ├── anfis_multisensor.pth   # Multi-sensor variant
│   └── anfis_multisensor_meta.json
│
├── training/                   # Training pipeline and datasets
│   ├── train_anfis_torch.py    # Model training script
│   ├── generate_dataset_v3.py  # Dataset generation from sensor logs
│   ├── label_dataset.py        # Dataset labeling utility
│   ├── labeled_dataset.csv     # Final labeled training dataset
│   └── anfis_training_dataset.csv
│
├── libs/                       # Local ANFIS library dependencies
│   ├── anfis-pytorch/          # Primary PyTorch ANFIS implementation
│   ├── anfis_package/          # Legacy version (reference)
│   └── anfis_patched/          # Raspberry Pi patched version
│
├── archive/                    # Older log files and superseded scripts
├── .gitignore
├── Pipfile
└── Pipfile.lock
```

---

## 🚀 How to Run

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
# or using pipenv
pipenv install
```

### Hardware required
- Raspberry Pi 4 Model B
- Neo-6M GPS module connected to `/dev/ttyAMA0`
- MPU-6050 IMU on I2C Bus 1
- BMP280 Barometer on I2C Bus 1

### Run the live supervisor
```bash
cd src
python master_supervisor_v3.py
```

### Train the model (optional)
```bash
cd training
python train_anfis_torch.py
```

---

## 📦 Dependencies

Key libraries used:
- `torch` — ANFIS model training and inference
- `pyserial` — GPS UART communication
- `smbus2` — I2C sensor communication (IMU, Barometer)
- `pynmea2` — NMEA GPS sentence parsing
- `pandas`, `numpy` — Data processing
- `scikit-learn` — Preprocessing and evaluation

---

## 🔬 Research

A research paper based on this work is currently **in preparation**, covering:
- ANFIS-based GPS spoofing detection methodology
- Multi-sensor fusion architecture for UAV trust evaluation
- Comparative analysis across 4 attack scenarios (Normal, Gradual Spoofing, Sudden Spoofing, Jamming)

---

## 👤 Author

**Bulbul Deora**  
B.Tech CSE (Cyber Security), Central University of Jammu  
📧 bulbul.adeora@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/bulbul-deora-6047092a9/) | [GitHub](https://github.com/Bulbul19)

---

## 📄 License

This project is licensed under the MIT License.
