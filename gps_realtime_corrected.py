#!/usr/bin/env python3
"""
gps_realtime_corrected.py

Reads NEO-6M GPS from /dev/ttyAMA0, computes features, normalizes using saved meta,
loads ANFIS model (anfis_3in1out.pth + anfis_3in1out_meta.json) and infers gps_trust_score.
Logs to gps_realtime_log.csv.

Third feature: distance jump (meters) between consecutive fixes is used as the third input.
If you have an IMU later, replace distance_jump with vibration_x easily.
"""

import time
import json
import math
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import serial
import pynmea2

# -----------------------------
# USER CONFIG
# -----------------------------
MODEL_PATH = "anfis_v3.pth"
META_PATH = "anfis_v3_meta.json"
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 9600
LOG_CSV = "gps_realtime_log.csv"
DEVICE = "cpu"
APPLY_SIGMOID = True   # map raw model output to 0..1
# -----------------------------

device = torch.device(DEVICE)

# -----------------------------
# ANFIS model class (must match training)
# -----------------------------
class ANFIS_3in_1out(nn.Module):
    def __init__(self, n_mfs: int = 5):
        super().__init__()
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** 3

        # MF banks (structure only — parameters will be loaded from saved state)
        # We need same attribute names as training: mf_x1, mf_x2, mf_x3, consequents
        # Create placeholder GaussianMF modules with dummy initial params (they'll be overwritten)
        class GaussianMF(nn.Module):
            def __init__(self, mean=0.0, sigma=1.0):
                super().__init__()
                self.mean = nn.Parameter(torch.tensor(float(mean)))
                self.log_sigma = nn.Parameter(torch.tensor(float(np.log(max(sigma, 1e-6)))))

            def forward(self, x):
                sigma = torch.exp(self.log_sigma)
                return torch.exp(-0.5 * ((x - self.mean) / sigma).pow(2))

        self.mf_x1 = nn.ModuleList([GaussianMF() for _ in range(n_mfs)])
        self.mf_x2 = nn.ModuleList([GaussianMF() for _ in range(n_mfs)])
        self.mf_x3 = nn.ModuleList([GaussianMF() for _ in range(n_mfs)])
        self.consequents = nn.Parameter(torch.randn(self.n_rules, 4) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

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

        x1_exp = x1.unsqueeze(1)
        x2_exp = x2.unsqueeze(1)
        x3_exp = x3.unsqueeze(1)

        # a[:,0]*x1 + a[:,1]*x2 + a[:,2]*x3 + a[:,3]
        y_rule = (a[:, 0].unsqueeze(0) * x1_exp +
                  a[:, 1].unsqueeze(0) * x2_exp +
                  a[:, 2].unsqueeze(0) * x3_exp +
                  a[:, 3].unsqueeze(0))  # [B, n_rules]

        y = (w_norm * y_rule).sum(dim=1)
        return y


# -----------------------------
# Helpers
# -----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    """Return distance in meters between two lat/lon pairs."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def safe_extract_hdop(gga_msg):
    """Try multiple attribute names to extract hdop."""
    for attr in ("hdop", "horizontal_dil", "horizontal_dilution"):
        if hasattr(gga_msg, attr):
            val = getattr(gga_msg, attr)
            if val not in (None, "", " "):
                try:
                    return float(val)
                except Exception:
                    pass
    # fallback: some messages include 'pdop' etc. Try .data if available (rare)
    try:
        if hasattr(gga_msg, "data") and len(gga_msg.data) > 8:
            # GGA fields: 8th field often HDOP index 8 (0-based 7) — be conservative
            v = gga_msg.data[8]
            return float(v)
    except Exception:
        pass
    return None


# -----------------------------
# Load model + meta
# -----------------------------
if not Path(MODEL_PATH).exists():
    raise SystemExit(f"Model file not found: {MODEL_PATH}")

if not Path(META_PATH).exists():
    raise SystemExit(f"Meta file not found: {META_PATH}")

with open(META_PATH, "r") as f:
    meta = json.load(f)

# meta expected keys: "mu" and "sigma" (lists)
if "mu" in meta:
    mu = np.array(meta["mu"], dtype=float)
    sigma = np.array(meta["sigma"], dtype=float)
elif "mean" in meta and "std" in meta:
    mu = np.array(meta["mean"], dtype=float)
    sigma = np.array(meta["std"], dtype=float)
else:
    raise SystemExit("Meta JSON must contain 'mu' and 'sigma' (or 'mean' and 'std').")

# instantiate model and load weights (must match saved architecture)
model = ANFIS_3in_1out(n_mfs=5)
state = torch.load(MODEL_PATH, map_location=device)
# If state is a dict with 'model_state' key, adapt; otherwise assume direct state_dict
if set(["consequents"]).issubset(set(state.keys())) or any(k.startswith("mf_x1") for k in state.keys()):
    # state is the state_dict saved from ANFIS_3in_1out
    model.load_state_dict(state)
else:
    # try if user saved the entire model
    try:
        model.load_state_dict(state)
    except Exception as e:
        raise SystemExit(f"Unable to load model state_dict: {e}")

model.eval()
print("Loaded ANFIS model and metadata successfully.")

# -----------------------------
# CSV log header
# -----------------------------
if not Path(LOG_CSV).exists():
    with open(LOG_CSV, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["ts", "lat", "lon", "sats", "hdop", "distance_jump_m", "trust_score"])

# -----------------------------
# Serial / GPS loop
# -----------------------------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

prev_lat = None
prev_lon = None
prev_ts = None

print("Listening on serial port:", SERIAL_PORT)
print("Press Ctrl-C to stop.")

try:
    while True:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue

        # Only parse GGA sentences (contains lat/lon, sats, hdop)
        if raw.startswith("$GNGGA") or raw.startswith("$GPGGA"):
            try:
                msg = pynmea2.parse(raw)
            except Exception as e:
                # ignore malformed sentence
                # print("Parse error:", e)
                continue

            # safe extraction
            sats = None
            try:
                if hasattr(msg, "num_sats"):
                    sats = int(msg.num_sats)
                elif hasattr(msg, "satellites_in_view"):
                    sats = int(msg.satellites_in_view)
            except Exception:
                sats = None

            hdop = safe_extract_hdop(msg)
            # coordinates come in as floats via pynmea2
            try:
                lat = float(msg.latitude)
                lon = float(msg.longitude)
            except Exception:
                # if parse failed for coordinates skip
                continue

            ts = time.time()

            # compute distance jump in meters
            distance_jump = 0.0
            if prev_lat is not None and prev_lon is not None and prev_ts is not None:
                try:
                    distance_jump = haversine_m(prev_lat, prev_lon, lat, lon)
                except Exception:
                    distance_jump = 0.0

            # update previous fix
            prev_lat, prev_lon, prev_ts = lat, lon, ts

            # If any feature missing, skip
            if sats is None or hdop is None:
                # print("Missing sats/hdop, skipping sample")
                continue

            # Build input vector: [hdop, satellites, distance_jump]
            # NOTE: training used ['hdop','satellites','vibration_x'] - using distance_jump as third feature proxy
            x_raw = np.array([hdop, sats, distance_jump], dtype=float)

            # normalize
            x_norm = (x_raw - mu) / sigma

            xt = torch.tensor(x_norm.reshape(1, -1), dtype=torch.float32).to(device)
            with torch.no_grad():
                y = model(xt).cpu().numpy().squeeze()
            if APPLY_SIGMOID:
                trust = 1.0 / (1.0 + math.exp(-float(y)))
            else:
                trust = float(y)
            trust = max(0.0, min(1.0, trust))

            # print nicely
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Lat:{lat:.6f} Lon:{lon:.6f} sats:{sats} hdop:{hdop:.2f} jump:{distance_jump:.2f}m trust:{trust:.3f}")

            # log
            with open(LOG_CSV, "a", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow([time.time(), lat, lon, sats, hdop, distance_jump, trust])

        # else ignore non-GGA lines (or you may handle GSA/GSV for extra info)
except KeyboardInterrupt:
    print("\nStopping realtime GPS supervisor.")
finally:
    try:
        ser.close()
    except Exception:
        pass
