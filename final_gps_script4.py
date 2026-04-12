#!/usr/bin/env python3
"""
master_supervisor.py

GPS -> ANFIS -> fuzzy fallback -> CSV logger pipeline.
"""

import time
import math
import csv
import json
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import serial
import pynmea2

# -----------------------------
# USER CONFIG
# -----------------------------
SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 9600
MODEL_PATH = "anfis_v3.pth"
META_PATH = "anfis_v3_meta.json"
CSV_LOG = "realtime_log.csv"
ANFIS_SIMILARITY_THRESHOLD = 3.0
VIBRATION_WINDOW = 8
DEVICE = "cpu"
MIN_SATS_TO_ACCEPT = 3
MAX_HDOP_ACCEPT = 50.0
# -----------------------------

# -----------------------------
# Utilities
# -----------------------------
def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_gga_line(line):
    try:
        msg = pynmea2.parse(line)
        if isinstance(msg, pynmea2.types.talker.GGA):
            gps_qual = int(getattr(msg, "gps_qual", 0))
            valid_fix = gps_qual > 0
            sats = int(getattr(msg, "num_sats", 0))
            hdop = float(getattr(msg, "horizontal_dil", getattr(msg, "hdop", 0.0)))
            lat = float(msg.latitude or 0.0)
            lon = float(msg.longitude or 0.0)
            alt = float(getattr(msg, "altitude", 0.0) or 0.0)
            return {"valid_fix": valid_fix, "lat": lat, "lon": lon, "sats": sats, "hdop": hdop, "altitude": alt}
    except Exception:
        return {"valid_fix": False}
    return {"valid_fix": False}

# -----------------------------
# ANFIS model classes
# -----------------------------
class GaussianMF(nn.Module):
    def __init__(self, mean: float, sigma: float):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(float(mean)))
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(max(sigma, 1e-6)))))
    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        return torch.exp(-0.5 * ((x - self.mean)/sigma).pow(2.0))

class ANFIS_3in_1out(nn.Module):
    def __init__(self, n_mfs: int = 3):
        super().__init__()
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** 3
        self.mf_x1 = nn.ModuleList([GaussianMF(0.0,1.0) for _ in range(n_mfs)])
        self.mf_x2 = nn.ModuleList([GaussianMF(0.0,1.0) for _ in range(n_mfs)])
        self.mf_x3 = nn.ModuleList([GaussianMF(0.0,1.0) for _ in range(n_mfs)])
        self.consequents = nn.Parameter(torch.randn(self.n_rules,4)*0.1)

    def forward(self, x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        m1 = torch.stack([mf(x1) for mf in self.mf_x1], dim=1)
        m2 = torch.stack([mf(x2) for mf in self.mf_x2], dim=1)
        m3 = torch.stack([mf(x3) for mf in self.mf_x3], dim=1)
        B = x.shape[0]
        rules = torch.zeros(B, self.n_rules, device=x.device)
        idx=0
        for i in range(self.n_mfs):
            for j in range(self.n_mfs):
                for k in range(self.n_mfs):
                    rules[:,idx] = m1[:,i]*m2[:,j]*m3[:,k]
                    idx += 1
        denom = rules.sum(dim=1, keepdim=True) + 1e-9
        w_norm = rules / denom
        a = self.consequents
        x1e = x1.unsqueeze(1); x2e = x2.unsqueeze(1); x3e = x3.unsqueeze(1)
        y_rule = a[:,0].unsqueeze(0)*x1e + a[:,1].unsqueeze(0)*x2e + a[:,2].unsqueeze(0)*x3e + a[:,3].unsqueeze(0)
        y = (w_norm * y_rule).sum(dim=1)
        return y

def load_anfis_model(model_path, meta_path, device="cpu"):
    meta = {}
    if Path(meta_path).exists():
        with open(meta_path,"r") as f:
            meta = json.load(f)
    n_mfs = meta.get("n_mfs",3)
    model = ANFIS_3in_1out(n_mfs)
    try:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict):
            model.load_state_dict(state)
        print("[INFO] ANFIS model loaded.")
    except Exception as e:
        print("[ERROR] ANFIS load failed:", e)
        model = None
    return model, meta

def is_similar_to_training(x_raw, mu, sigma, threshold=ANFIS_SIMILARITY_THRESHOLD):
    diff = x_raw - mu
    sig = np.copy(sigma); sig[sig==0]=1.0
    dist = np.sqrt(np.sum((diff/sig)**2))
    return float(dist), dist <= threshold

# -----------------------------
# Robust fuzzy fallback
# -----------------------------
def fuzzy_trust(sats, hdop, jump, vib):
    sats = float(max(0.0, min(20.0, sats)))
    hdop = float(max(0.0, min(10.0, hdop if hdop is not None else 10.0)))
    jump = float(max(0.0, min(100.0, jump)))
    vib = float(max(0.0, min(5.0, vib)))  # assuming vibration range 0–5 m/s²

    # --- Memberships ---
    # Satellites
    sat_few = max(0, min(1, (6-sats)/6))
    sat_ok  = max(0, min((sats-4)/4, (12-sats)/4))
    sat_many = max(0, min(1, (sats-10)/10))
    
    # HDOP
    hd_ex = max(0, min(1, (1.5-hdop)/0.7))
    hd_good = max(0, min((hdop-1.2)/0.8, (3-hdop)/1))
    hd_poor = max(0, min((hdop-2.5)/7.5,1))
    
    # Jump
    j_norm = max(0, min(1, (5-jump)/5))
    j_imp  = max(0, min(1, (jump-20)/80))
    
    # Vibration
    vib_low  = max(0, min(1, (0.2-vib)/0.2))
    vib_med  = max(0, min((vib-0.1)/0.2, (0.5-vib)/0.2))
    vib_high = max(0, min(1, (vib-0.4)/0.6))

    # --- Rules ---
    rules = []
    rules.append((max(sat_few, hd_poor), 0))                  # BAD due to poor sats/HDOP
    rules.append((j_imp, 0))                                   # BAD due to jump
    rules.append((min(sat_ok, hd_good, j_norm, vib_low), 1))   # OK
    rules.append((min(sat_many, hd_ex, j_norm, vib_low), 2))   # GOOD

    xs = np.linspace(0,1,101)
    out_agg = np.zeros_like(xs)

    for act, idx in rules:
        if act <= 0: continue
        if idx == 0: a,b,c = 0,0,0.45
        elif idx == 1: a,b,c = 0.35,0.6,0.8
        else: a,b,c = 0.7,1,1
        mf_vals = np.array([max(0,min(1,(c-x)/(c-b))) if x>b else max(0,min(1,(x-a)/(b-a) if b-a!=0 else 0)) for x in xs])
        out_agg = np.maximum(out_agg, np.minimum(mf_vals, act))

    if out_agg.sum() == 0: return 0.05  # minimum fallback to avoid zero
    return float(np.clip((xs*out_agg).sum()/out_agg.sum(), 0,1))

def map_score_to_label(score):
    if score>=0.70: return "GOOD"
    if score>=0.45: return "JAMMED/WEAK"
    return "SPOOFED"

def ensure_csv_header(path):
    header = ["timestamp","lat","lon","sats","hdop","jump","vib","anfis_score","fuzzy_score","final_score","final_label"]
    p = Path(path)
    if not p.exists():
        with open(p,"w",newline="") as f:
            csv.writer(f).writerow(header)

# -----------------------------
# Main loop
# -----------------------------
def main():
    print("Starting GPS -> ANFIS -> fuzzy -> CSV pipeline...")
    model, meta = load_anfis_model(MODEL_PATH, META_PATH, DEVICE)
    mu = np.array(meta.get("mu")) if meta.get("mu") is not None else None
    sigma = np.array(meta.get("sigma")) if meta.get("sigma") is not None else None

    ensure_csv_header(CSV_LOG)
    recent_jumps = deque(maxlen=VIBRATION_WINDOW)
    last_lat = last_lon = last_time = None

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print("[ERROR] Could not open serial port:", e)
        return

    try:
        while True:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw: continue
            parsed = parse_gga_line(raw)
            if not parsed.get("valid_fix",False): continue

            lat, lon = parsed["lat"], parsed["lon"]
            sats, hdop = parsed["sats"], parsed["hdop"]
            altitude = parsed.get("altitude",0.0)
            if sats < MIN_SATS_TO_ACCEPT: continue

            # compute jump
            tnow = time.time()
            jump_m = 0.0
            if last_lat is not None:
                dt = max(1e-6, tnow-last_time)
                jump_m = haversine_m(last_lat,last_lon,lat,lon)
            last_lat,last_lon,last_time = lat,lon,tnow
            recent_jumps.append(jump_m)
            vib_x = float(np.std(list(recent_jumps))) if recent_jumps else 0.0

            # ANFIS
            raw_input = np.array([hdop if hdop is not None else 99.0, float(sats), float(vib_x)])
            anfis_score = None
            similar = False
            if model is not None and mu is not None and sigma is not None:
                sigma_arr = np.copy(sigma); sigma_arr[sigma_arr==0]=1.0
                dist, similar = is_similar_to_training(raw_input, mu, sigma_arr)
                if similar:
                    xnorm = (raw_input - mu)/sigma_arr
                    xt = torch.tensor(xnorm.reshape(1,-1),dtype=torch.float32,device=DEVICE)
                    with torch.no_grad():
                        raw_out = model(xt).cpu().numpy().squeeze()
                        anfis_score = float(1/(1+math.exp(-float(raw_out))))

            # Fuzzy fallback
            fuzzy_score = fuzzy_trust(sats, hdop if hdop is not None else 99.0, min(jump_m,100.0), vib_x)

            # Fusion logic
            if anfis_score is not None:
                if similar:
                    final_score = 0.8*anfis_score + 0.2*fuzzy_score
                else:
                    final_score = 0.2*anfis_score + 0.8*fuzzy_score
            else:
                final_score = fuzzy_score

            final_label = map_score_to_label(final_score)

            # CSV
            row = [ now_ts(), round(lat,6), round(lon,6), int(sats), round(hdop,2),
                    round(jump_m,3), round(vib_x,3),
                    round(float(anfis_score) if anfis_score is not None else 0.0,4),
                    round(float(fuzzy_score),4),
                    round(float(final_score),4),
                    final_label ]
            with open(CSV_LOG,"a",newline="") as f:
                csv.writer(f).writerow(row)

            # Console
            print(f"[{row[0]}] Lat:{lat:.6f} Lon:{lon:.6f} sats:{sats} hdop:{hdop:.2f} "
                  f"jump:{jump_m:.2f}m vib:{vib_x:.2f} anfis:{row[7]:.3f} "
                  f"fuzzy:{row[8]:.3f} final:{row[9]:.3f} => {final_label}")

    except KeyboardInterrupt:
        print("\nExiting (Ctrl-C).")
    finally:
        try: ser.close()
        except Exception: pass

if __name__ == "__main__":
    main()
