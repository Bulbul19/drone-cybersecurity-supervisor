#!/usr/bin/env python3
"""
final_gps_script6.py
Hybrid Supervisor: GPS + Barometer -> ANFIS (80%) + Fuzzy (20%) -> Decision
UPDATED: Uses external barometer_driver.Barometer
"""

import serial
import time
import csv
import math
import pickle
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# -----------------------------
# BAROMETER DRIVER (YOUR DRIVER)
# -----------------------------
from barometer_driver import Barometer

# -----------------------------
# PYTORCH / ANFIS CHECK
# -----------------------------
try:
    import torch
    from anfis import AnfisNet
    TORCH_AVAILABLE = True
except Exception:
    print("[WARN] PyTorch/ANFIS not available → Fuzzy only mode")
    TORCH_AVAILABLE = False

# -----------------------------
# USER CONFIG
# -----------------------------
SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 115200
CSV_LOG = "realtime_log.csv"
ANFIS_MODEL_PATH = "anfis_model.pkl"

# -----------------------------
# FUZZY LOGIC
# -----------------------------
def tri(x, a, b, c):
    x = float(x)
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b: return (x - a) / (b - a)
    return (c - x) / (c - b)

def fuzzy_trust_score(sats, hdop, jump, alt_diff):

    sat_few  = tri(sats, 0, 0, 6)
    sat_ok   = tri(sats, 4, 8, 12)
    sat_many = tri(sats, 10, 20, 20)

    hd_good = tri(hdop, 0, 1, 2)
    hd_ok   = tri(hdop, 1.5, 3, 4)
    hd_bad  = tri(hdop, 3, 10, 10)

    vib_low  = tri(jump, 0, 0, 5)
    vib_med  = tri(jump, 3, 10, 20)
    vib_high = tri(jump, 15, 100, 100)

    alt_match = tri(alt_diff, 0, 0, 5)
    alt_susp  = tri(alt_diff, 3, 10, 15)
    alt_div   = tri(alt_diff, 10, 100, 100)

    fire_low  = max(sat_few, hd_bad, vib_high, alt_div)
    fire_med  = max(min(sat_ok, hd_ok), alt_susp)
    fire_high = min(sat_many, hd_good, vib_low, alt_match)

    total = fire_low + fire_med + fire_high
    if total == 0:
        return 0.5

    return (fire_low * 0.2 + fire_med * 0.6 + fire_high * 1.0) / total

# -----------------------------
# HELPERS
# -----------------------------
def get_distance_metres(lat1, lon1, lat2, lon2):
    R = 6371000.0
    x = math.radians(lon2 - lon1) * math.cos(0.5 * math.radians(lat1 + lat2))
    y = math.radians(lat2 - lat1)
    return R * math.sqrt(x*x + y*y)

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_anfis(path):
    if not TORCH_AVAILABLE:
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
            model.eval()
        print("[INFO] ANFIS model loaded")
        return model
    except Exception as e:
        print(f"[WARN] ANFIS load failed: {e}")
        return None

# -----------------------------
# MAIN
# -----------------------------
def main():

    # ---- SERIAL ----
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"[ERROR] Serial failed: {e}")
        return

    # ---- LOAD MODELS ----
    anfis_model = load_anfis(ANFIS_MODEL_PATH)

    # ---- BAROMETER INIT (BEFORE LOOP) ----
    try:
        my_baro = Barometer()
        print("[INFO] Barometer initialized")
    except Exception as e:
        print(f"[WARN] Barometer init failed: {e}")
        my_baro = None

    # ---- CSV INIT ----
    if not Path(CSV_LOG).exists():
        with open(CSV_LOG, "w") as f:
            f.write("timestamp,sats,hdop,vib,alt_gps,alt_baro,alt_diff,anfis,fuzzy,final,action\n")

    print("[INFO] Supervisor running...")
    print("-" * 60)

    last_lat = last_lon = None
    last_time = time.time()
    gps_offset = baro_offset = 0.0
    calibrated = False

    while True:
        try:
            line = ser.readline().decode("ascii", errors="replace").strip()

            if "GGA" not in line:
                continue

            parts = line.split(",")
            if len(parts) < 10 or parts[6] == "0":
                continue

            lat_raw = parts[2]
            lon_raw = parts[4]
            sats = float(parts[7])
            hdop = float(parts[8]) if parts[8] else 1.0
            gps_alt = float(parts[9])

            # ---- BAROMETER READ ----
            if my_baro:
                try:
                    baro_alt = my_baro.get_altitude()
                    print(f"Barometer Altitude: {baro_alt:.2f} m")
                except:
                    baro_alt = 0.0
            else:
                baro_alt = 0.0

            # ---- CALIBRATION ----
            if not calibrated:
                gps_offset = gps_alt
                baro_offset = baro_alt
                calibrated = True
                print("[INFO] Altitude calibrated")

            rel_gps = gps_alt - gps_offset
            rel_baro = baro_alt - baro_offset
            alt_diff = abs(rel_gps - rel_baro)

            # ---- VIBRATION ----
            curr_time = time.time()
            vib = 0.0

            lat = float(lat_raw[:2]) + float(lat_raw[2:]) / 60
            lon = float(lon_raw[:3]) + float(lon_raw[3:]) / 60

            if last_lat is not None:
                dist = get_distance_metres(last_lat, last_lon, lat, lon)
                dt = curr_time - last_time
                if dt > 0:
                    vib = dist / dt

            last_lat, last_lon, last_time = lat, lon, curr_time

            # ---- ANFIS ----
            anfis_val = 0.0
            if anfis_model:
                try:
                    input_t = torch.tensor([[min(sats,13), max(hdop,1), min(vib,50)]],
                                           dtype=torch.float32)
                    with torch.no_grad():
                        anfis_val = float(anfis_model(input_t))
                        anfis_val = max(0.0, min(1.0, anfis_val))
                except:
                    anfis_val = 0.0

            # ---- FUZZY ----
            fuzzy_val = fuzzy_trust_score(sats, hdop, vib, alt_diff)

            # ---- FUSION ----
            final_score = (0.8 * anfis_val + 0.2 * fuzzy_val) if anfis_model else fuzzy_val

            if final_score < 0.45:
                action = "SPOOFED"
            elif final_score < 0.725:
                action = "JAMMED/WEAK"
            else:
                action = "GOOD"

            # ---- LOG ----
            log = f"{now_ts()},{sats},{hdop},{vib:.3f},{rel_gps:.1f},{rel_baro:.1f},{alt_diff:.1f},{anfis_val:.3f},{fuzzy_val:.3f},{final_score:.3f},{action}"
            print(log)

            with open(CSV_LOG, "a") as f:
                f.write(log + "\n")

        except Exception:
            pass

# -----------------------------
if __name__ == "__main__":
    main()
