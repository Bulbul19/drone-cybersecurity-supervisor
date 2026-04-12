#!/usr/bin/env python3
"""
final_gps_with_imu_fuzzy.py

GPS -> ANFIS -> fuzzy (with IMU) -> fusion -> CSV logger

- Serial: /dev/serial0 @ 9600
- MODEL_PATH / META_PATH should point to your ANFIS files (anfis_v3.pth, anfis_v3_meta.json)
- If scikit-fuzzy is installed, a full Mamdani engine is used. Otherwise a smooth heuristic fallback is used.
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
import smbus2
import board
import busio
import adafruit_bme280.basic as adafruit_bme280

# Try import skfuzzy (optional)
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except Exception:
    SKFUZZY_AVAILABLE = False

# -----------------------------
# USER CONFIG
# -----------------------------
SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 9600
MODEL_PATH = "anfis_multisensor.pth"
META_PATH = "anfis_multisensor_meta.json"
CSV_LOG = "realtime_log_8.csv"
ANFIS_SIMILARITY_THRESHOLD = 3.0
VIBRATION_WINDOW = 8
DEVICE = "cpu"
MIN_SATS_TO_ACCEPT = 3
MAX_HDOP_ACCEPT = 50.0
print("[DEBUG] MODEL_PATH =", MODEL_PATH)
print("[DEBUG] META_PATH  =", META_PATH)
# -----------------------------
MPU_ADDR = 0x68
BME_ADDR = 0x76
I2C_BUS = 1
# ==============================
# I2C
# ==============================
bus = smbus2.SMBus(I2C_BUS)
# -----------------------------
# Utilities
# -----------------------------
def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_gga_line(line):
    """Return dict with valid_fix, lat, lon, sats, hdop, altitude"""
    try:
        msg = pynmea2.parse(line)
    except Exception:
        return {"valid_fix": False}

    try:
        if isinstance(msg, pynmea2.types.talker.GGA) or line.startswith(
            ("$GNGGA", "$GPGGA", "$GAGGA", "$GLGGA")
        ):
            gps_qual = int(getattr(msg, "gps_qual", 0) or 0)
            valid_fix = gps_qual > 0

            sats = int(
                getattr(msg, "num_sats",
                        getattr(msg, "num_sv", 0)) or 0
            )

            hdop_raw = getattr(msg, "horizontal_dil", None) or getattr(msg, "hdop", None)
            hdop = float(hdop_raw) if hdop_raw not in (None, "") else None

            lat = float(msg.latitude or 0.0)
            lon = float(msg.longitude or 0.0)
            alt = float(getattr(msg, "altitude", 0.0) or 0.0)

            if hdop is not None:
                if math.isnan(hdop) or hdop <= 0 or hdop > MAX_HDOP_ACCEPT:
                    hdop = None

            return {
                "valid_fix": valid_fix,
                "lat": lat,
                "lon": lon,
                "sats": sats,
                "hdop": hdop,
                "altitude": alt,
            }
    except Exception:
        pass

    return {"valid_fix": False}


# ==============================
# MPU6050
# ==============================
MPU_ADDR = 0x68
prev_accel_mag = None

def init_mpu():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0x00)  # wake up
    bus.write_byte_data(MPU_ADDR, 0x1C, 0x00)  # ±2g
    bus.write_byte_data(MPU_ADDR, 0x1B, 0x00)  # ±250 dps
    time.sleep(0.1)
    print("[INFO] MPU6050 initialized")


def read_word(reg):
    h = bus.read_byte_data(MPU_ADDR, reg)
    l = bus.read_byte_data(MPU_ADDR, reg + 1)
    v = (h << 8) + l
    return v - 65536 if v > 32768 else v


def read_mpu():
    global prev_accel_mag

    ax = read_word(0x3B) / 16384.0
    ay = read_word(0x3D) / 16384.0
    az = read_word(0x3F) / 16384.0

    gx = read_word(0x43) / 131.0
    gy = read_word(0x45) / 131.0
    gz = read_word(0x47) / 131.0

    accel_mag = math.sqrt(ax*ax + ay*ay + az*az)

    if prev_accel_mag is None:
        accel_vib = 0.0
    else:
        accel_vib = abs(accel_mag - prev_accel_mag)

    prev_accel_mag = accel_mag

    gyro_mag = abs(gx) + abs(gy) + abs(gz)

    return accel_vib, gyro_mag

# ==============================
# BME280
# ==============================
bme = None   # global

def init_bme280():
    global bme
    i2c = busio.I2C(board.SCL, board.SDA)
    bme = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)

    # IMPORTANT: set local sea level pressure
    bme.sea_level_pressure = 1023.0  # change if you know local value

    print("[INFO] BME280 initialized")
# ==============================
# BAROMETER HELPER
# ==============================
# ==============================
# BME280 (RAW – SAFE MODE)
# ==============================
def get_baro_altitude():
    try:
        alt = float(bme.altitude)

        # sanity check
        if alt < -50 or alt > 2000:
            return None

        return alt

    except Exception as e:
        print("[WARN] BME280 read failed:", e)
        return None

# -----------------------------
# ANFIS model (same architecture used when saving)
# -----------------------------
class GaussianMF(nn.Module):
    def __init__(self, mean: float, sigma: float):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        self.log_sigma = nn.Parameter(
            torch.log(torch.tensor(max(sigma, 1e-6), dtype=torch.float32))
        )

    def forward(self, x):
        sigma = torch.exp(self.log_sigma) + 1e-9
        return torch.exp(-0.5 * ((x - self.mean) / sigma) ** 2)
GaussMf = GaussianMF
class ANFIS_Nin_1out(nn.Module):
    def __init__(self, n_inputs: int, n_mfs: int = 2):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_inputs * n_mfs

        self.mf_layers = nn.ModuleList()
        for _ in range(n_inputs):
            mfs = nn.ModuleList()
            centers = torch.linspace(-1.5, 1.5, n_mfs)
            for c in centers:
                mfs.append(GaussMf(c.item(), sigma=1.0))
            self.mf_layers.append(mfs)

        self.consequents = nn.Parameter(
            torch.randn(self.n_rules, n_inputs + 1) * 0.05
        )

    def forward(self, X):
        B = X.shape[0]
        rule_acts = []

        for i in range(self.n_inputs):
            xi = X[:, i]
            for mf in self.mf_layers[i]:
                rule_acts.append(mf(xi))

        W = torch.stack(rule_acts, dim=1)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-6)

        X_ext = torch.cat([X, torch.ones(B, 1, device=X.device)], dim=1)
        Y_rules = X_ext @ self.consequents.T

        return (W * Y_rules).sum(dim=1)
def load_anfis_model(model_path, meta_path, device="cpu"):
    if not Path(meta_path).exists():
        print("[WARN] ANFIS meta not found")
        return None, None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    mu = meta["mu"]
    sigma = meta["sigma"]
    n_inputs = len(mu)
    n_mfs = meta["n_mfs"]

    print(f"[INFO] Loading ANFIS: inputs={n_inputs}, mfs={n_mfs}, rules={n_inputs * n_mfs}")

    model = ANFIS_Nin_1out(
        n_inputs=n_inputs,
        n_mfs=n_mfs
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("[INFO] Loaded ANFIS multisensor model")

    return model, meta

def is_similar_to_training(x_raw, mu, sigma, threshold=ANFIS_SIMILARITY_THRESHOLD):
    if mu is None or sigma is None:
        return float("inf"), False

    x_raw = np.asarray(x_raw, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if x_raw.shape[0] != mu.shape[0]:
        print("[WARN] ANFIS similarity check skipped (dimension mismatch)")
        return float("inf"), False

    sig = np.copy(sigma)
    sig[sig == 0.0] = 1.0

    z = (x_raw - mu) / sig
    dist = float(np.linalg.norm(z))

    return dist, (dist <= threshold)

# -----------------------------
# Fuzzy engine builder (skfuzzy) - 5 inputs: satellites, hdop, jump, max_accel, mean_gyro
# -----------------------------
def build_fuzzy_engine():
    satellites = ctrl.Antecedent(np.arange(0,21,1), 'satellites')
    hdop       = ctrl.Antecedent(np.arange(0,21,0.1), 'hdop')
    jump       = ctrl.Antecedent(np.arange(0,101,1), 'jump') # meters
    accel_vib = ctrl.Antecedent(np.arange(0,2.01,0.01), 'accel_vib')
    gyro_rate = ctrl.Antecedent(np.arange(0,301,1), 'gyro_rate')
    max_accel = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'max_accel')
    mean_gyro = ctrl.Antecedent(np.arange(0, 301, 1), 'mean_gyro')
    alt_err   = ctrl.Antecedent(np.arange(0,51,0.5), 'alt_err')
    trust     = ctrl.Consequent(np.arange(0,1.01,0.01), 'trust')

    # Satellites
    satellites['FEW']  = fuzz.trimf(satellites.universe,[0,0,6])
    satellites['OKAY'] = fuzz.trimf(satellites.universe,[4,8,12])
    satellites['MANY'] = fuzz.trimf(satellites.universe,[10,20,20])

    # HDOP
    hdop['EXCELLENT'] = fuzz.trimf(hdop.universe,[0,0.7,1.5])
    hdop['GOOD']      = fuzz.trimf(hdop.universe,[1.2,2.0,3.0])
    hdop['POOR']      = fuzz.trimf(hdop.universe,[2.5,21,21])

    # Jump
    jump['NORMAL']      = fuzz.trimf(jump.universe,[0,0,5])
    jump['SUSPICIOUS']  = fuzz.trimf(jump.universe,[4,10,30])
    jump['IMPOSSIBLE']  = fuzz.trimf(jump.universe,[20,100,100])

    # Acceleration vibration
    accel_vib['LOW']  = fuzz.trimf(accel_vib.universe,[0,0,0.1])
    accel_vib['MED']  = fuzz.trimf(accel_vib.universe,[0.05,0.2,0.4])
    accel_vib['HIGH'] = fuzz.trimf(accel_vib.universe,[0.3,2.0,2.0])

    # Gyro
    gyro_rate['STABLE'] = fuzz.trimf(gyro_rate.universe,[0,0,15])
    gyro_rate['TURN']   = fuzz.trimf(gyro_rate.universe,[10,60,120])
    gyro_rate['FAST']   = fuzz.trimf(gyro_rate.universe,[100,300,300])

    # Altitude error
    alt_err['SMALL'] = fuzz.trimf(alt_err.universe,[0,0,3])
    alt_err['MED']   = fuzz.trimf(alt_err.universe,[2,6,10])
    alt_err['LARGE'] = fuzz.trimf(alt_err.universe,[8,50,50])
   # acceleration
    max_accel['LOW'] = fuzz.trimf(max_accel.universe, [0, 0, 0.8])
    max_accel['MEDIUM'] = fuzz.trimf(max_accel.universe, [0.5, 1.5, 3.0])
    max_accel['HIGH'] = fuzz.trimf(max_accel.universe, [2.5, 5.0, 5.0])
   # gyro 
    mean_gyro['STABLE'] = fuzz.trimf(mean_gyro.universe, [0, 0, 20])
    mean_gyro['TURNING'] = fuzz.trimf(mean_gyro.universe, [15, 60, 150])
    mean_gyro['FAST_TURN'] = fuzz.trimf(mean_gyro.universe, [100, 300, 300])
    # Trust
    trust['LOW']    = fuzz.trimf(trust.universe,[0,0,0.4])
    trust['MEDIUM'] = fuzz.trimf(trust.universe,[0.3,0.6,0.8])
    trust['HIGH']   = fuzz.trimf(trust.universe,[0.7,1.0,1.0])

    rules = [
    # HARD GPS QUALITY LIMITS
    ctrl.Rule(hdop['POOR'], trust['LOW']),
    ctrl.Rule(satellites['FEW'], trust['LOW']),

    # IMPOSSIBLE PHYSICS
    ctrl.Rule(jump['IMPOSSIBLE'], trust['LOW']),
    ctrl.Rule(max_accel['HIGH'] & mean_gyro['FAST_TURN'], trust['LOW']),

    # WEAK GPS BUT CONSISTENT MOTION
    ctrl.Rule(
        satellites['OKAY'] &
        hdop['GOOD'] &
        jump['NORMAL'],
        trust['MEDIUM']
    ),

    # STRONG GPS + CONSISTENT IMU
    ctrl.Rule(
        satellites['MANY'] &
        hdop['EXCELLENT'] &
        jump['NORMAL'] &
        max_accel['LOW'] &
        mean_gyro['STABLE'],
        trust['HIGH']
    ),
   ctrl.Rule(
    satellites['OKAY'] &
    hdop['EXCELLENT'] &
    jump['NORMAL'] &
    max_accel['LOW'] &
    mean_gyro['STABLE'] &
    alt_err['LARGE'],
    trust['MEDIUM']
   ),
   ctrl.Rule(
    alt_err['LARGE'] &
    max_accel['LOW'] &
    mean_gyro['STABLE'],
    trust['LOW']
   ),
]

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# -----------------------------
# Fuzzy heuristic fallback (never returns 0 for reasonable inputs)
# -----------------------------
def fuzzy_heuristic(sats, hdop, jump, max_accel, mean_gyro):
    """
    Smooth fallback combining GPS and IMU when skfuzzy not installed.
    Returns trust in [0,1], never exactly 0 for realistic values.
    Balanced: GPS and IMU contribute equally.
    """
    # Normalize inputs to 0..1 (higher -> better for sats, lower hdop -> better)
    sat_score = np.clip((sats / 12.0), 0.0, 1.0)            # 0..1
    hd_score = np.clip(1.0 - (hdop / 6.0), 0.0, 1.0)        # lower hdop => higher score
    jump_score = np.clip(1.0 - (jump / 10.0), 0.0, 1.0)     # small jump => high score
    gps_score = 0.5 * (sat_score + hd_score) * 1.0 * jump_score + 0.0

    # IMU scores: low accel & low gyro => good
    accel_score = np.clip(1.0 - (max_accel / 3.0), 0.0, 1.0)    # 0..1
    gyro_score = np.clip(1.0 - (mean_gyro / 120.0), 0.0, 1.0)   # 0..1
    imu_score = 0.5 * (accel_score + gyro_score)

    # Balanced fusion: average and then soften (avoid exact zeros)
    combined = 0.5 * gps_score + 0.5 * imu_score
    # smooth floor so typical noise doesn't produce exactly 0
    combined = float(np.clip(combined, 0.02, 1.0))
    return combined

# -----------------------------
# Combined fuzzy wrapper (calls skfuzzy sim if available else heuristic)
# -----------------------------
def compute_fuzzy_score(sim, sats, hdop, jump_m, max_accel_mag, mean_gyro_mag):
    # normalize hdop fallback
    hdop_in = float(hdop if hdop is not None else 99.0)
    if sim is not None:
        try:
            sim.input['satellites'] = float(sats)
            sim.input['hdop'] = float(hdop_in)
            sim.input['jump'] = float(min(jump_m, 100.0))
            sim.input['max_accel'] = float(max_accel_mag)
            sim.input['mean_gyro'] = float(mean_gyro_mag)
            sim.compute()
            # trust exists
            score = float(sim.output.get('trust', 0.0))
            # clamp
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            # fall back to heuristic
            # print("[WARN] fuzzy sim compute error:", e)
            pass
    # fallback
    return fuzzy_heuristic(sats, hdop_in, jump_m, max_accel_mag, mean_gyro_mag)

# -----------------------------
# CSV helper
# -----------------------------
CSV_HEADER = [
    "timestamp",

    # GPS
    "lat",
    "lon",
    "sats",
    "hdop",
    "gps_alt",

    # Motion / kinematics
    "jump",            # GPS position jump (m)
    "accel_vib",       # |accel_mag − 1g|
    "gyro_mag",        # sum or mean gyro (deg/s)

    # Barometer
    "baro_alt",
    "alt_err",         # |gps_alt − baro_alt|

    # Decision system
    "anfis",
    "fuzzy",
    "final_score",
    "final_label"
]
def ensure_csv_header(path):
    p = Path(path)
    if not p.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def map_score_to_label(score):
    if score >= 0.70:
        return "GOOD"
    if score >= 0.45:
        return "JAMMED/WEAK"
    return "SPOOFED"

# -----------------------------
# Placeholder IMU feature function
# -----------------------------
def compute_imu_features():
    """
    Return (max_accel_mag, mean_gyro_mag).
    - max_accel_mag in G (e.g., 0..5)
    - mean_gyro_mag in deg/s (e.g., 0..300)
    Replace this implementation with your IMU read + window processing.
    """
    # === Simple placeholder ===
    # If you have IMU stream, replace this block with code that reads IMU,
    # maintains a short buffer and returns the required features.
    return 0.0, 0.0
def write_csv_row(
    timestamp,
    lat, lon, sats, hdop, gps_alt,
    jump_m, accel_vib, gyro_mag,
    baro_alt, alt_err,
    anfis_score, fuzzy_score, final_score, final_label
):
    row = [
        timestamp,

        # GPS
        round(lat, 6) if lat is not None else "",
        round(lon, 6) if lon is not None else "",
        int(sats) if sats is not None else "",
        round(hdop, 2) if hdop is not None else "",
        round(gps_alt, 2) if gps_alt is not None else "",

        # IMU
        round(jump_m, 3),
        round(accel_vib, 4),
        round(gyro_mag, 2),

        # Barometer
        round(baro_alt, 2) if baro_alt is not None else "",
        round(alt_err, 2),

        # Decision
        round(anfis_score, 4) if anfis_score is not None else "",
        round(fuzzy_score, 4),
        round(final_score, 4),
        final_label
    ]

    with open(CSV_LOG, "a", newline="") as f:
        csv.writer(f).writerow(row)

# -----------------------------
# Main loop
# -----------------------------
def main():
    print("Starting GPS -> ANFIS -> fuzzy -> CSV pipeline...")
    model, meta = load_anfis_model(MODEL_PATH, META_PATH, DEVICE)
    mu = np.array(meta.get("mu")) if meta.get("mu") is not None else None
    sigma = np.array(meta.get("sigma")) if meta.get("sigma") is not None else None
    print("Loaded mf_layers sizes:")
    for i, layer in enumerate(model.mf_layers):
        print(f"Input {i}: {len(layer)} MFs")

    fuzzy_sim = build_fuzzy_engine() if SKFUZZY_AVAILABLE else None
    ensure_csv_header(CSV_LOG)

    baro_offset = None
    baro_calibrated = False
    recent_jumps = deque(maxlen=VIBRATION_WINDOW)
    last_lat = last_lon = last_time = None
    last_baro_alt = None
    bme = None
    init_bme280()   
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print("[ERROR] Could not open serial port:", e)
        return
    init_mpu()

    try:
        while True:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue

            parsed = parse_gga_line(raw)
            if not parsed.get("valid_fix", False):
                # silently ignore invalid/no-fix lines (no spam)
                continue

            lat = parsed["lat"]; lon = parsed["lon"]
            sats = int(parsed["sats"]); hdop = parsed["hdop"]
            altitude = parsed.get("altitude", 0.0)
            if sats < MIN_SATS_TO_ACCEPT:
                continue

            # compute jump
            tnow = time.time()
            jump_m = 0.0
            if last_lat is not None and last_lon is not None and last_time is not None:
                dt = max(1e-6, tnow - last_time)
                jump_m = haversine_m(last_lat, last_lon, lat, lon)
            last_lat, last_lon, last_time = lat, lon, tnow

            recent_jumps.append(jump_m)
            vib_x = float(np.std(list(recent_jumps))) if len(recent_jumps) > 0 else 0.0

            # ---- IMU READ ----
            try:
                accel_vib, gyro_mag = read_mpu()
            except Exception as e:
                print("[WARN] MPU read failed:", e)
                accel_vib = 0.0
                gyro_mag = 0.0
            # ---- BAROMETER ----
            # -------------------------------
# BAROMETER READ
# -------------------------------
            baro_alt = get_baro_altitude()

            if baro_alt is None:
                baro_alt = last_baro_alt
            else:
                last_baro_alt = baro_alt


# -------------------------------
# BARO OFFSET CALIBRATION (RUN ONCE)
# Uses barometer ONLY (no GPS)
# -------------------------------
#            if not baro_calibrated and baro_alt is not None:
#               baro_offset = baro_alt
#               baro_calibrated = True
#               print(f"[INFO] Barometer calibrated | offset = {baro_offset:.2f} m")


# -------------------------------
# ALTITUDE ERROR COMPUTATION
# -------------------------------
#            if (
 #               baro_calibrated
  #              and altitude is not None
   #             and baro_alt is not None
    #        ):
#                baro_alt_corr = baro_alt - baro_offset
 #               alt_err = abs(altitude - baro_alt_corr)
  #          else:
   #             alt_err = None
            if altitude is not None and baro_alt is not None:
                alt_err = abs(altitude - baro_alt)
            else:
                alt_err = None
            # Prepare ANFIS input: user used [hdop, sats, vibration] for model training
            # --- Correct 6-input ANFIS vector (MUST match training order) ---
            raw_input = np.array([
               float(sats),
               float(hdop) if hdop is not None else 10.0,
               float(jump_m),
               float(alt_err),
               float(accel_vib),
               float(gyro_mag),
            ], dtype=float)

            anfis_score = None

            if model is not None and mu is not None and sigma is not None:
                try:
                    mu_arr = np.asarray(mu, dtype=float)
                    sigma_arr = np.asarray(sigma, dtype=float)
                    sigma_arr[sigma_arr == 0.0] = 1.0

        # Similarity check (soft gate)
                    dist, similar = is_similar_to_training(raw_input, mu_arr, sigma_arr)

        # Normalize exactly like training
                    xnorm = (raw_input - mu_arr) / sigma_arr
                    xt = torch.tensor(xnorm.reshape(1, -1), dtype=torch.float32, device=DEVICE)

                    with torch.no_grad():
                       anfis_out = model(xt).item()

        # ANFIS already outputs trust ∈ [0,1]
                    anfis_score = float(np.clip(anfis_out, 0.0, 1.0))

        # Optional confidence damping
                    if not similar:
                        anfis_weight = 0.25   # still used, but weak
                    else:
                        anfis_weight = 0.6    # trusted more when similar

                except Exception as e:
                    anfis_score = None
                if alt_err > 50.0:
                      anfis_weight *= 0.5
                force_bad = False

                if sats <= 5:
                 force_bad = True

                if hdop >= 2.5:
                 force_bad = True

                if alt_err is not None and alt_err >= 30:
                 force_bad = True

            # Fuzzy (skfuzzy sim if available else heuristic) - includes IMU inputs
            fuzzy_score = compute_fuzzy_score(fuzzy_sim, sats, hdop, min(jump_m, 100.0), accel_vib, gyro_mag)

            # Fusion: balanced (if ANFIS available use it equally with fuzzy)
            if anfis_score is not None:
               fuzzy_weight = 1.0 - anfis_weight
               final_score = anfis_weight * anfis_score + fuzzy_weight * fuzzy_score
               final_source = "ANFIS+FUZZY"
            else:
               final_score = float(fuzzy_score)
               final_source = "FUZZY_ONLY"

            if force_bad:
               final_score = min(final_score, 0.45)
               final_label = "BAD"
               final_source = "RULE_OVERRIDE"
            else:      
               final_label = map_score_to_label(final_score)

            # CSV log
            write_csv_row(
                timestamp = now_ts(),

                lat = lat,
                lon = lon,
                sats = sats,
                hdop = hdop,
                gps_alt = altitude,

                jump_m = jump_m,
                accel_vib = accel_vib,
                gyro_mag = gyro_mag,

                baro_alt = baro_alt,
                alt_err = alt_err,

                anfis_score = anfis_score,
                fuzzy_score = fuzzy_score,
                final_score = final_score,
                final_label = map_score_to_label(final_score)

            )
            row = [
               now_ts(),                                  # timestamp
               round(float(lat), 6),                      # lat
               round(float(lon), 6),                      # lon
               int(sats),                                 # satellites
               round(float(hdop) if hdop is not None else 99.0, 2),  # hdop
               round(float(altitude) if altitude is not None else 0.0, 1),  # gps_alt

               round(float(jump_m), 3),                   # jump (m)
               round(float(accel_vib), 3),                # IMU accel vibration
               round(float(gyro_mag), 2),                 # IMU gyro magnitude

               round(float(baro_alt) if baro_alt is not None else -1.0, 1),
               round(float(alt_err) if alt_err is not None else 0.0, 2),
               round(float(anfis_score) if anfis_score is not None else 0.0, 4),
               round(float(fuzzy_score), 4),
               round(float(final_score), 4),
               final_label
            ]
            try:
                with open(CSV_LOG, "a", newline="") as f:
                    csv.writer(f).writerow(row)
            except Exception as e:
                print("[WARN] CSV write:", e)

            # Console print (clean)
            print(
                f"[{row[0]}] "
                f"Lat:{row[1]:.6f} Lon:{row[2]:.6f} "
                f"sats:{row[3]} hdop:{row[4]:.2f} "
                f"gpsAlt:{row[5]:.1f}m "
                f"jump:{row[6]:.3f}m "
                f"accVib:{row[7]:.3f}g "
                f"gyro:{row[8]:.2f}dps "
                f"baroAlt:{row[9]:.1f}m "
                f"altErr:{row[10]:.2f}m "
                f"anfis:{row[11]:.3f} "
                f"fuzzy:{row[12]:.3f} " 
                f"final:{row[13]:.3f} => {row[14]} ({final_source})"
            )    
    except KeyboardInterrupt:
        print("\nExiting (Ctrl-C).")
    finally:
        try:
            ser.close()
        except Exception:

            pass

if __name__ == "__main__":
    main()
