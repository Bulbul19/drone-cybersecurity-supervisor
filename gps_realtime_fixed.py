#!/usr/bin/env python3
"""
gps_realtime_fixed.py

Production-ready realtime GPS → ANFIS inference script.

- Auto-detects ANFIS training n_mfs from the saved state dict (avoids shape mismatch).
- Loads mu/sigma (normalization) from meta JSON (fields: 'mu' and 'sigma').
- Reads NMEA (GGA) from SERIAL_PORT and computes hdop, satellites, distance jump.
- Builds input vector [hdop, satellites, vibration_x] and runs model.
- Prints timestamped output and scenario classification.
- Optionally logs to CSV.

Edit USER CONFIG below to match your system.
"""

import json
import time
import math
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import serial
import pynmea2

# -----------------------------
# USER CONFIG
# -----------------------------
MODEL_PATH = "anfis_v3.pth"                 # saved state_dict (training output)
META_PATH = "anfis_v3_meta.json"            # saved mu/sigma and optionally n_mfs
SERIAL_PORT = "/dev/serial0"                # or "/dev/ttyAMA0"
BAUD_RATE = 9600
ENABLE_LOGGING = True
LOG_PATH = "realtime_log.csv"
# -----------------------------

device = torch.device("cpu")


# -----------------------------
# Helper: haversine distance (meters)
# -----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# -----------------------------
# Define the model class exactly as training (GaussianMF + ANFIS)
# This must match keys in saved state_dict (consequents and mf_x*.mean/log_sigma)
# -----------------------------
class GaussianMF(nn.Module):
    def __init__(self, mean=0.0, sigma=1.0):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(float(mean)))
        # training used log_sigma param; keep same name "log_sigma" if present in checkpoint
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(max(sigma, 1e-6)))))

    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        return torch.exp(-0.5 * ((x - self.mean) / sigma).pow(2))


class ANFIS_3in_1out(nn.Module):
    def __init__(self, n_mfs: int = 3):
        super().__init__()
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** 3

        # create MF ModuleLists so keys match checkpoint: mf_x1.<i>.mean, mf_x1.<i>.log_sigma, etc.
        self.mf_x1 = nn.ModuleList([GaussianMF() for _ in range(n_mfs)])
        self.mf_x2 = nn.ModuleList([GaussianMF() for _ in range(n_mfs)])
        self.mf_x3 = nn.ModuleList([GaussianMF() for _ in range(n_mfs)])

        # consequent parameters saved as "consequents" in checkpoint shape [n_rules, 4]
        self.consequents = nn.Parameter(torch.randn(self.n_rules, 4) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3] -> x1 (hdop), x2 (sats), x3 (vibration)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        m1 = torch.stack([mf(x1) for mf in self.mf_x1], dim=1)  # [B, n_mfs]
        m2 = torch.stack([mf(x2) for mf in self.mf_x2], dim=1)
        m3 = torch.stack([mf(x3) for mf in self.mf_x3], dim=1)

        B = x.shape[0]
        rule_fire = torch.zeros(B, self.n_rules, device=x.device)

        idx = 0
        for i in range(self.n_mfs):
            for j in range(self.n_mfs):
                for k in range(self.n_mfs):
                    rule_fire[:, idx] = m1[:, i] * m2[:, j] * m3[:, k]
                    idx += 1

        denom = rule_fire.sum(dim=1, keepdim=True) + 1e-9
        w_norm = rule_fire / denom  # [B, n_rules]

        a = self.consequents  # [n_rules, 4]
        # compute a1*x1 + a2*x2 + a3*x3 + b for each rule (broadcast)
        x1_exp = x1.unsqueeze(1)  # [B,1]
        x2_exp = x2.unsqueeze(1)
        x3_exp = x3.unsqueeze(1)

        y_rule = (a[:, 0].unsqueeze(0) * x1_exp +
                  a[:, 1].unsqueeze(0) * x2_exp +
                  a[:, 2].unsqueeze(0) * x3_exp +
                  a[:, 3].unsqueeze(0))  # [B, n_rules]

        y = (w_norm * y_rule).sum(dim=1)  # [B]
        # NOTE: training saved raw outputs; we apply sigmoid externally to map to [0,1]
        return y


# -----------------------------
# Load model with autodetection of n_mfs
# -----------------------------
def load_model_and_meta(model_path: str, meta_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    # load state dict first (no strict)
    state = torch.load(model_path, map_location="cpu")

    # if saved state is directly a state_dict or full model, handle common cases:
    if isinstance(state, dict) and any(k.startswith("mf_x1") for k in state.keys()):
        state_dict = state
    else:
        # if the file contains only state_dict under key 'state_dict' or similar:
        # try common keys:
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            # worst case: maybe the file is the raw state_dict itself
            state_dict = state

    # determine n_mfs by counting mf_x1.<i> keys
    n_mfs = 0
    for key in state_dict.keys():
        # key examples: "mf_x1.0.mean", "mf_x1.0.log_sigma"
        if key.startswith("mf_x1."):
            # extract index between first '.' and next '.'
            parts = key.split(".")
            if len(parts) >= 3:
                try:
                    idx = int(parts[1])
                    n_mfs = max(n_mfs, idx + 1)
                except Exception:
                    pass

    if n_mfs == 0:
        # fallback: try to infer from "consequents" size
        if "consequents" in state_dict:
            conseq = state_dict["consequents"]
            if hasattr(conseq, "shape"):
                n_rules = conseq.shape[0]
                # n_rules = n_mfs**3  => n_mfs = round(n_rules ** (1/3))
                n_mfs = int(round(n_rules ** (1.0/3.0)))
        if n_mfs == 0:
            n_mfs = 3  # default fallback

    # build model with inferred n_mfs
    model = ANFIS_3in_1out(n_mfs=n_mfs)
    # load state dict into model (use strict=False to allow meta differences)
    model.load_state_dict(state_dict, strict=False)

    # load meta json
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Accept both 'mu'/'sigma' (what we used) or 'mean'/'std' variants
    if "mu" in meta and "sigma" in meta:
        mu = np.array(meta["mu"], dtype=float)
        sigma = np.array(meta["sigma"], dtype=float)
    elif "mean" in meta and "std" in meta:
        mu = np.array(meta["mean"], dtype=float)
        sigma = np.array(meta["std"], dtype=float)
    else:
        # try other names
        mu = np.array(meta.get("mean") or meta.get("mu") or [0.0, 0.0, 0.0], dtype=float)
        sigma = np.array(meta.get("std") or meta.get("sigma") or [1.0, 1.0, 1.0], dtype=float)

    sigma[sigma == 0] = 1.0
    return model.eval(), mu, sigma


# -----------------------------
# Map raw jump -> vibration_x (training distribution approx)
# Explanation:
# - During dataset generation you used vibration_x in range approx [2,8].
# - Live GPS jump (meters) is small (0.x). To produce reasonable inputs,
#   we map jump_m -> vibration_x by a small linear scaling + offset, clipped
#   into [0.5, 8.0]. Tweak SCALE/OFFSET to match your IMU if available.
# -----------------------------
VIB_SCALE = 5.0
VIB_OFFSET = 3.0
def map_jump_to_vibration(jump_m):
    vib = jump_m * VIB_SCALE + VIB_OFFSET
    vib = float(max(0.5, min(8.0, vib)))
    return vib


# -----------------------------
# Scenario classification (tunable)
# -----------------------------
def classify_scenario(score):
    if score >= 0.8:
        return "GOOD GPS"
    if score >= 0.45:
        return "JAMMED/WEAK"
    return "SPOOFED/UNRELIABLE"


# -----------------------------
# Safe HDOP extraction
# -----------------------------
def extract_hdop(nmea_msg):
    for attr in ("hdop", "horizontal_dil", "horizontal_dilution"):
        if hasattr(nmea_msg, attr):
            val = getattr(nmea_msg, attr)
            try:
                if val is None or val == "":
                    continue
                return float(val)
            except Exception:
                continue
    return None


# -----------------------------
# Main realtime loop
# -----------------------------
def main():
    print("Realtime ANFIS GPS supervisor starting...")
    try:
        model, mu, sigma = load_model_and_meta(MODEL_PATH, META_PATH)
    except Exception as e:
        print("❌ ERROR loading model:", e)
        return

    print("Loaded ANFIS model and normalization metadata successfully.")
    print("Listening on serial port:", SERIAL_PORT)
    print("Press Ctrl-C to stop.\n")

    # prepare CSV logger if requested
    if ENABLE_LOGGING:
        log_file = open(LOG_PATH, "a", newline="")
        csvw = csv.writer(log_file)
        # write header if file created new
        if os.stat(LOG_PATH).st_size == 0:
            csvw.writerow(["timestamp", "lat", "lon", "sats", "hdop", "jump_m", "vibration_x", "gps_trust_score", "scenario"])

    # serial open
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print("ERROR opening serial port:", e)
        return

    prev_lat = None
    prev_lon = None
    prev_time = None

    try:
        while True:
            try:
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue

                # process only GGA (position, hdop, sat count)
                if "GGA" not in raw:
                    continue

                try:
                    msg = pynmea2.parse(raw)
                except Exception:
                    # malformed sentence
                    continue

                # gps quality and data
                try:
                    sats = int(msg.num_sats) if msg.num_sats not in (None, "", "0") else 0
                except Exception:
                    sats = 0

                hdop_val = extract_hdop(msg)
                if hdop_val is None:
                    # fallback: treat as poor
                    hdop_val = 10.0

                # lat/lon parsed by pynmea2 as floats
                try:
                    lat = float(msg.latitude)
                    lon = float(msg.longitude)
                except Exception:
                    # skip if invalid coordinates
                    continue

                tstamp = time.time()
                # distance jump meters between successive fixes
                jump_m = 0.0
                if prev_lat is not None and prev_lon is not None:
                    try:
                        jump_m = haversine_m(prev_lat, prev_lon, lat, lon)
                    except Exception:
                        jump_m = 0.0

                prev_lat, prev_lon, prev_time = lat, lon, tstamp

                # build vibration_x either from IMU (if available) or map jump -> vib
                vibration_x = map_jump_to_vibration(jump_m)

                # prepare input vector in same order as training
                raw_input = np.array([hdop_val, float(sats), float(vibration_x)], dtype=float)

                # normalize using saved mu/sigma
                x_norm = (raw_input - mu) / sigma
                x_tensor = torch.tensor(x_norm.reshape(1, -1), dtype=torch.float32)

                # forward pass
                with torch.no_grad():
                    raw_output = model(x_tensor).cpu().numpy().squeeze()
                # map raw output to [0,1] using sigmoid (training used raw outputs)
                trust_score = 1.0 / (1.0 + np.exp(-float(raw_output)))

                scenario = classify_scenario(trust_score)

                tstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{tstr}] Lat:{lat:.6f} Lon:{lon:.6f} sats:{sats} hdop:{hdop_val:.2f} jump:{jump_m:.2f}m vib:{vibration_x:.2f} trust:{trust_score:.3f} => {scenario}")

                if ENABLE_LOGGING:
                    csvw.writerow([tstr, lat, lon, sats, hdop_val, f"{jump_m:.3f}", f"{vibration_x:.3f}", f"{trust_score:.4f}", scenario])
                    log_file.flush()

            except KeyboardInterrupt:
                print("Interrupted by user — exiting.")
                break
            except Exception as e:
                # don't crash on single parse failure
                print("Runtime error (ignored):", e)

    finally:
        try:
            ser.close()
        except Exception:
            pass
        if ENABLE_LOGGING:
            log_file.close()


if __name__ == "__main__":
    main()
