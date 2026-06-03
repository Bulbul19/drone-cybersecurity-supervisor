#!/usr/bin/env python3
# gps_realtime_v3.py
# Realtime supervisor — loads ANFIS model and reads NMEA from serial

import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import serial
import pynmea2
from typing import Tuple

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "anfis_v3.pth"
META_PATH = "anfis_v3_meta.json"
SERIAL_PORT = "/dev/serial0"   # change to /dev/ttyAMA0 if needed
BAUD_RATE = 9600
SMOOTH_ALPHA = 0.3  # smoothing for live trust score
# -----------------------------


# Use the exact same model class as saved by train_anfis2.py
class GaussMf(nn.Module):
    def __init__(self, centers: np.ndarray, sigmas: np.ndarray):
        super().__init__()
        centers = np.array(centers, dtype=float)
        sigmas = np.array(sigmas, dtype=float)
        self.register_parameter("mean", nn.Parameter(torch.tensor(centers, dtype=torch.float32)))
        self.register_parameter("log_sigma", nn.Parameter(torch.log(torch.tensor(sigmas, dtype=torch.float32))))

    def forward(self, x):
        mean = self.mean.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0)
        x = x.unsqueeze(1)
        return torch.exp(-0.5 * ((x - mean) / (sigma + 1e-9)) ** 2)


class ANFIS_3in_1out(nn.Module):
    def __init__(self, n_mfs: int = 3):
        super().__init__()
        self.n_mfs = int(n_mfs)
        self.n_rules = self.n_mfs ** 3
        self.mf_x1 = nn.ModuleList()
        self.mf_x2 = nn.ModuleList()
        self.mf_x3 = nn.ModuleList()
        for _ in range(self.n_mfs):
            # initialize with placeholder params; they will be replaced during load_state_dict
            self.mf_x1.append(GaussMf([0.0], [1.0]))
            self.mf_x2.append(GaussMf([0.0], [1.0]))
            self.mf_x3.append(GaussMf([0.0], [1.0]))
        self.consequents = nn.Parameter(torch.randn(self.n_rules, 4) * 0.1)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        m1 = torch.cat([m(x1) for m in self.mf_x1], dim=1)
        m2 = torch.cat([m(x2) for m in self.mf_x2], dim=1)
        m3 = torch.cat([m(x3) for m in self.mf_x3], dim=1)

        B = x.shape[0]
        rules = []
        for i in range(self.n_mfs):
            for j in range(self.n_mfs):
                for k in range(self.n_mfs):
                    rules.append(m1[:, i] * m2[:, j] * m3[:, k])
        rule_fire = torch.stack(rules, dim=1)
        denom = rule_fire.sum(dim=1, keepdim=True) + 1e-9
        w = rule_fire / denom

        a = self.consequents
        x1e = x1.unsqueeze(1)
        x2e = x2.unsqueeze(1)
        x3e = x3.unsqueeze(1)
        y_rule = a[:, 0].unsqueeze(0) * x1e + a[:, 1].unsqueeze(0) * x2e + a[:, 2].unsqueeze(0) * x3e + a[:, 3].unsqueeze(0)
        y = (w * y_rule).sum(dim=1)
        return y


# -----------------------------
# Helpers
# -----------------------------
def safe_load_meta(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load mu,sigma and n_mfs from meta; handle older key names."""
    with open(path, "r") as f:
        meta = json.load(f)
    # accept either 'mu'/'sigma' or 'mean'/'std' or 'mean'/'sigma'
    if "mu" in meta:
        mu = np.array(meta["mu"], dtype=float)
    elif "mean" in meta:
        mu = np.array(meta["mean"], dtype=float)
    else:
        raise KeyError("meta must contain 'mu' or 'mean' entries")

    if "sigma" in meta:
        sigma = np.array(meta["sigma"], dtype=float)
    elif "std" in meta:
        sigma = np.array(meta["std"], dtype=float)
    else:
        # fallback small std to avoid div0
        sigma = np.ones_like(mu) * 1.0

    n_mfs = int(meta.get("n_mfs", -1))
    return mu, sigma, n_mfs


def infer_n_mfs_from_checkpoint(path: str) -> int:
    """If meta lacks n_mfs, inspect checkpoint keys to infer it."""
    state = torch.load(path, map_location="cpu")
    # count mf_x1.*.mean keys
    keys = list(state.keys())
    # pattern: 'mf_x1.X.mean' or 'mf_x1.X.weight' depending on model; search for 'mf_x1.' occurrences
    counts = sum(1 for k in keys if k.startswith("mf_x1.") and ".mean" in k)
    if counts > 0:
        return counts
    # other possibility: 'mf_x1.0.mean' etc
    counts2 = len([k for k in keys if k.startswith("mf_x1.")])
    if counts2 > 0:
        # may include log_sigma etc; compute unique indices
        indices = set()
        for k in keys:
            if k.startswith("mf_x1."):
                # format mf_x1.{i}.mean
                parts = k.split(".")
                if len(parts) >= 3:
                    try:
                        indices.add(int(parts[1]))
                    except Exception:
                        pass
        if indices:
            return max(indices) + 1
    # fallback: try to infer from consequents size (n_rules = conseq_rows)
    if "consequents" in state:
        conseq_shape = state["consequents"].shape
        n_rules = conseq_shape[0]
        # find integer n_mfs s.t. n_mfs**3 == n_rules
        for n in range(2, 20):
            if n ** 3 == n_rules:
                return n
    return -1


def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def classify_score(score: float) -> str:
    # tune thresholds as needed
    if score >= 0.75:
        return "GOOD GPS"
    if score >= 0.45:
        return "JAMMED/WEAK"
    return "SPOOFED"


# -----------------------------
# Main loader + realtime loop
# -----------------------------
def main():
    # load meta
    try:
        mu, sigma, n_mfs = safe_load_meta(META_PATH)
    except Exception as e:
        print("ERROR loading meta:", e)
        return

    # if n_mfs not present, infer from checkpoint
    if n_mfs <= 0:
        n_mfs = infer_n_mfs_from_checkpoint(MODEL_PATH)
        if n_mfs <= 0:
            print("Could not infer n_mfs from checkpoint; please retrain and include 'n_mfs' in meta.")
            return
        print("Inferred n_mfs =", n_mfs)

    # instantiate model and load state
    model = ANFIS_3in_1out(n_mfs=n_mfs)
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        # If checkpoint is a full model dict (state_dict) or state
        if isinstance(state, dict) and any(k.startswith("mf_x1.") for k in state.keys()):
            model.load_state_dict(state)
        else:
            # maybe file contains state_dict under 'model_state' or similar
            if "model_state" in state:
                model.load_state_dict(state["model_state"])
            elif "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                # assume state is state_dict
                model.load_state_dict(state)
    except Exception as e:
        print("ERROR loading model:", e)
        return

    model.eval()
    print("Loaded ANFIS v3 model and meta successfully.")
    print("Listening on serial port:", SERIAL_PORT)

    # open serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print("ERROR opening serial port:", e)
        return

    prev_lat = None
    prev_lon = None
    prev_time = None
    smooth_trust = None

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                # keep looping
                continue

            # parse NMEA only when GGA or GNRMC present
            if line.startswith("$GPGGA") or line.startswith("$GNGGA") or line.startswith("$GNRMC"):
                try:
                    msg = pynmea2.parse(line)
                except Exception:
                    continue

                # Extract sats, hdop, lat, lon
                try:
                    lat = float(msg.latitude)
                    lon = float(msg.longitude)
                except Exception:
                    # skip invalid GPS parses
                    continue

                sats = getattr(msg, "num_sats", None) or getattr(msg, "num_sat", None) or getattr(msg, "satellites", None)
                try:
                    sats = int(sats)
                except Exception:
                    sats = 0

                # hdop extraction - handle different attribute names
                hdop = None
                for attr in ("hdop", "horizontal_dil", "horizontal_dilution"):
                    if hasattr(msg, attr):
                        try:
                            hdop = float(getattr(msg, attr))
                            break
                        except Exception:
                            pass
                if hdop is None:
                    # fallback to estimation
                    hdop = 5.0

                # compute jump in meters
                ts = time.time()
                jump_m = 0.0
                if prev_lat is not None:
                    jump_m = haversine_meters(prev_lat, prev_lon, lat, lon)
                prev_lat, prev_lon, prev_time = lat, lon, ts

                # --------------------------
                # get vibration_x: if you have IMU, read actual vib RMS. If not, approximate from NMEA or set 0.
                # We'll try to parse custom sentences "VIB" else fallback to small value
                # --------------------------
                vib = 0.0
                # Example: if you had IMU data stream, plug here
                # For now, use small default so that feature scale matches training
                vib = 3.0  # default baseline if you don't have IMU. Replace with real IMU read.

                # Prepare input vector (hdop, sats, vibration_x) and normalize using saved mu/sigma
                raw = np.array([hdop, sats, vib], dtype=float)
                Xn = (raw - mu) / (sigma + 1e-9)
                xt = torch.tensor(Xn.reshape(1, -1), dtype=torch.float32)

                with torch.no_grad():
                    raw_out = model(xt).cpu().numpy().squeeze().item() if hasattr(model(xt), "cpu") else float(model(xt).numpy().squeeze())
                # apply sigmoid to map to (0,1)
                trust = 1.0 / (1.0 + math.exp(-float(raw_out)))

                # smoothing
                if smooth_trust is None:
                    smooth_trust = trust
                else:
                    smooth_trust = SMOOTH_ALPHA * trust + (1 - SMOOTH_ALPHA) * smooth_trust

                scenario = classify_score(smooth_trust)

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Lat:{lat:.6f} Lon:{lon:.6f} sats:{sats} hdop:{hdop:.2f} jump:{jump_m:.2f}m vib:{vib:.2f} raw:{raw_out:.3f} trust:{smooth_trust:.3f} => {scenario}")

            # continue loop
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
