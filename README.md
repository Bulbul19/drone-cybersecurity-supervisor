# 🚁 AI-Based Drone Cybersecurity Supervisor

## 📌 Overview
This project implements a real-time **multi-sensor trust evaluation system** for drones using:
- GPS data
- IMU sensor data
- ANFIS-based AI model

The system detects anomalies and potential cyber threats in UAV operations.

---

## 🧠 System Pipeline

GPS + IMU → Sensor Fusion → ANFIS Model → Trust Score → Supervisor Decision

---

## ⚙️ Features
- Real-time GPS monitoring
- IMU-based motion validation
- Fuzzy logic + ANFIS fusion
- Trust-based anomaly detection

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/main.py
