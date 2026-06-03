#!/usr/bin/env python3
"""
master_supervisor_v2.py

Integrated Pipeline:
 - GPS NMEA from /dev/ttyAMA0
 - IMU (MPU6050) with Auto-Calibration (Tare)
 - Barometer (BME280) for relative altitude
 - ANFIS/Fuzzy logic for GPS Trust
 - Fuzzy Fusion for Altitude (GPS + Baro)
 - CSV Logging
"""

import time
import math
import csv
import json
import shutil
import os
from pathlib import Path
from collections import deque
from datetime import datetime, date

import numpy as np
import smbus2

# --- Optional Libraries ---
try:
    import serial
except ImportError:
    print("Error: pyserial not installed.")
    serial = None

try:
    import pynmea2
except ImportError:
    pynmea2 = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

try:
    import bme280
except ImportError:
    print("Error: RPi.bme280 not installed. Barometer will fail.")
    bme280 = None

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False

# -----------------------------
# CONFIGURATION
# -----------------------------
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 9600
MODEL_PATH = "anfis_v3.pth"
META_PATH = "anfis_v3_meta.json"
CSV_LOG = "realtime_log_v2.csv"
BACKUP_DIR = "backups"
ANFIS_SIMILARITY_THRESHOLD = 3.0
IMU_WINDOW_SIZE = 50
I2C_BUS = 1
MPU6050_ADDR = 0x68
BME280_ADDR = 0x76
DEVICE = "cpu"
MIN_SATS = 3

# -----------------------------
# UNIFIED SENSOR MANAGER
# Handles MPU6050 (w/ Calibration) and BME280
# -----------------------------
class SensorManager:
    def __init__(self, bus_number=1):
        self.bus = smbus2.SMBus(bus_number)
        
        # --- BME280 Setup ---
        self.baro_addr = BME280_ADDR
        self.ground_pressure = 1013.25
        self.last_baro_time = time.time()
        self.last_baro_alt = 0.0
        self.cal_params = None
        self.baro_available = False

        if bme280:
            try:
                self.cal_params = bme280.load_calibration_params(self.bus, self.baro_addr)
                self.baro_available = True
                print(f"[SENSORS] Barometer detected at {hex(self.baro_addr)}")
            except Exception as e:
                print(f"[WARN] Barometer init failed: {e}")

        # --- MPU6050 Setup ---
        self.mpu_addr = MPU6050_ADDR
        self.resting_g = 1.0
        self.imu_available = False

        try:
            # Wake up & Config to +/- 2g
            self.bus.write_byte_data(self.mpu_addr, 0x6B, 0)
            self.bus.write_byte_data(self.mpu_addr, 0x1C, 0) # Accel Config
            self.bus.write_byte_data(self.mpu_addr, 0x1B, 0) # Gyro Config
            self.imu_available = True
            print(f"[SENSORS] MPU6050 detected at {hex(self.mpu_addr)}")
            
            # Auto-Calibrate IMU
            print("[SENSORS] Calibrating IMU (Keep Still)...")
            self.calibrate_imu()
            
            # Set Ground Pressure
            if self.baro_available:
                self.set_ground_pressure()

        except Exception as e:
            print(f"[WARN] MPU6050 init failed: {e}")

    def calibrate_imu(self):
        """Tare the accelerometer to find resting gravity."""
        total = 0
        samples = 50
        for _ in range(samples):
            data = self.read_raw_imu()
            total += data['total_g']
            time.sleep(0.01)
        self.resting_g = total / samples
        print(f"[SENSORS] Calibration Complete. Resting G: {self.resting_g:.3f}")

    def set_ground_pressure(self):
        """Average pressure readings to set 0m altitude."""
        if not self.baro_available: return
        total = 0
        count = 10
        for _ in range(count):
            d = bme280.sample(self.bus, self.baro_addr, self.cal_params)
            total += d.pressure
            time.sleep(0.05)
        self.ground_pressure = total / count
        print(f"[SENSORS] Ground Pressure: {self.ground_pressure:.2f} hPa")

    def read_raw_imu(self):
        """Reads raw bytes and returns vectors."""
        if not self.imu_available:
            return {'total_g': 1.0, 'gyro_mag': 0.0, 'ax':0, 'ay':0, 'az':0, 'gx':0, 'gy':0, 'gz':0}
        
        try:
            # Block read 14 bytes
            data = self.bus.read_i2c_block_data(self.mpu_addr, 0x3B, 14)
            
            def to_signed16(high, low):
                v = (high << 8) | low
                return v - 65536 if v > 32768 else v

            ax = to_signed16(data[0], data[1]) / 16384.0
            ay = to_signed16(data[2], data[3]) / 16384.0
            az = to_signed16(data[4], data[5]) / 16384.0
            gx = to_signed16(data[8], data[9]) / 131.0
            gy = to_signed16(data[10], data[11]) / 131.0
            gz = to_signed16(data[12], data[13]) / 131.0

            total_g = math.sqrt(ax*ax + ay*ay + az*az)
            gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)
            
            return {
                'total_g': total_g, 'gyro_mag': gyro_mag,
                'ax': ax, 'ay': ay, 'az': az, 
                'gx': gx, 'gy': gy, 'gz': gz
            }
        except:
             return {'total_g': 1.0, 'gyro_mag': 0.0, 'ax':0, 'ay':0, 'az':0, 'gx':0, 'gy':0, 'gz':0}

    def get_data(self):
        """Returns consolidated dictionary of all sensor data."""
        # 1. IMU
        raw_imu = self.read_raw_imu()
        vibration = abs(self.resting_g - raw_imu['total_g'])
        
        # 2. Barometer
        alt = 0.0
        velocity_var = 0.0
        
        if self.baro_available:
            try:
                d = bme280.sample(self.bus, self.baro_addr, self.cal_params)
                # Hypsometric Formula
                alt = 44330 * (1.0 - pow(d.pressure / self.ground_pressure, 0.1903))
                
                # Velocity Variance
                t_now = time.time()
                dt = t_now - self.last_baro_time
                if dt > 0:
                    velocity_var = abs(alt - self.last_baro_alt) / dt
                
                self.last_baro_alt = alt
                self.last_baro_time = t_now
            except:
                pass

        return {
            'imu_vib': vibration,
            'imu_gyro': raw_imu['gyro_mag'],
            'raw_accel_mag': raw_imu['total_g'], # For ANFIS
            'baro_alt': alt,
            'baro_var': velocity_var
        }

# -----------------------------
# UTILS & LOGGING
# -----------------------------
def now_ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ensure_dir(path): os.makedirs(path, exist_ok=True)

CSV_HEADER = [
    "timestamp","lat","lon","gps_alt","baro_alt","fused_alt",
    "sats","hdop","jump","vib","final_trust","mode"
]

def ensure_csv(path):
    if not Path(path).exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_gga(line):
    if not line or "$G" not in line: return {"fix": False}
    try:
        if pynmea2:
            msg = pynmea2.parse(line.strip())
            if isinstance(msg, pynmea2.types.talker.GGA) and msg.gps_qual > 0:
                return {
                    "fix": True, "lat": msg.latitude, "lon": msg.longitude,
                    "sats": int(msg.num_sats), "hdop": float(msg.horizontal_dil),
                    "alt": float(msg.altitude)
                }
    except: pass
    return {"fix": False}

# -----------------------------
# FUZZY / ANFIS CLASSES (Preserved)
# -----------------------------
class GaussianMF(nn.Module):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(float(mean)))
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(max(sigma, 1e-6)))))
    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.mean) / torch.exp(self.log_sigma)).pow(2.0))

class ANFIS_3in_1out(nn.Module):
    def __init__(self, n_mfs=3):
        super().__init__()
        self.n_mfs = n_mfs; self.n_rules = n_mfs**3
        self.mf_x1 = nn.ModuleList([GaussianMF(0,1) for _ in range(n_mfs)])
        self.mf_x2 = nn.ModuleList([GaussianMF(0,1) for _ in range(n_mfs)])
        self.mf_x3 = nn.ModuleList([GaussianMF(0,1) for _ in range(n_mfs)])
        self.consequents = nn.Parameter(torch.randn(self.n_rules,4)*0.1)
    def forward(self, x):
        m1 = torch.stack([mf(x[:,0]) for mf in self.mf_x1], dim=1)
        m2 = torch.stack([mf(x[:,1]) for mf in self.mf_x2], dim=1)
        m3 = torch.stack([mf(x[:,2]) for mf in self.mf_x3], dim=1)
        rules = []
        for i in range(self.n_mfs):
            for j in range(self.n_mfs):
                for k in range(self.n_mfs):
                    rules.append(m1[:,i]*m2[:,j]*m3[:,k])
        w = torch.stack(rules, dim=1)
        w_norm = w / (w.sum(dim=1, keepdim=True)+1e-9)
        y_rule = (self.consequents[:,0]*x[:,0].unsqueeze(1) + 
                  self.consequents[:,1]*x[:,1].unsqueeze(1) + 
                  self.consequents[:,2]*x[:,2].unsqueeze(1) + 
                  self.consequents[:,3])
        return (w_norm * y_rule).sum(dim=1)

def load_anfis(path, meta_path):
    if not torch or not Path(path).exists(): return None, {}
    try:
        with open(meta_path) as f: meta = json.load(f)
        model = ANFIS_3in_1out(meta.get("n_mfs", 3))
        sd = torch.load(path, map_location="cpu")
        if "model_state_dict" in sd: sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()
        return model, meta
    except: return None, {}

def build_fuzzy_engine():
    if not SKFUZZY_AVAILABLE: return None
    # (Simplified for brevity, similar to your original code)
    sats = ctrl.Antecedent(np.arange(0, 21, 1), 'sats')
    hdop = ctrl.Antecedent(np.arange(0, 21, 0.1), 'hdop')
    trust = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'trust')
    sats['low'] = fuzz.trimf(sats.universe, [0,0,6])
    sats['high'] = fuzz.trimf(sats.universe, [10,20,20])
    hdop['good'] = fuzz.trimf(hdop.universe, [0,1,2])
    hdop['bad'] = fuzz.trimf(hdop.universe, [2,20,20])
    trust['low'] = fuzz.trimf(trust.universe, [0,0,0.5])
    trust['high'] = fuzz.trimf(trust.universe, [0.5,1,1])
    rules = [
        ctrl.Rule(sats['low'] | hdop['bad'], trust['low']),
        ctrl.Rule(sats['high'] & hdop['good'], trust['high'])
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def simple_fuzzy_fallback(sats, hdop):
    # Basic fallback if skfuzzy is missing
    score = 0.5
    if sats > 10: score += 0.3
    elif sats < 5: score -= 0.3
    if hdop < 1.5: score += 0.2
    elif hdop > 5: score -= 0.3
    return max(0.0, min(1.0, score))

# -----------------------------
# ALTITUDE FUSION LOGIC
# -----------------------------
def fuse_altitude(gps_alt, gps_trust, baro_alt, baro_var, imu_vib):
    """
    Combines GPS and Barometer based on reliability.
    """
    # 1. Barometer Trust
    # High variance (jumping) or High IMU vibration reduces Baro trust
    baro_trust = 1.0
    if baro_var > 2.0: baro_trust = 0.5
    if baro_var > 5.0: baro_trust = 0.1
    
    # If high vibration, physical sensor readings are suspect
    if imu_vib > 0.3: baro_trust *= 0.5
    
    # 2. Weighted Fusion
    total_trust = gps_trust + baro_trust
    if total_trust == 0: return baro_alt # Fallback
    
    fused = (gps_alt * gps_trust + baro_alt * baro_trust) / total_trust
    return fused

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("--- Supervisor v2: GPS + Baro + Calibrated IMU ---")
    ensure_dir(BACKUP_DIR)
    ensure_csv(CSV_LOG)
    
    # 1. Hardware Init
    sensors = SensorManager(bus_number=I2C_BUS)
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except:
        print("Serial Error")
        return

    # 2. AI Init
    model, meta = load_anfis(MODEL_PATH, META_PATH)
    fuzzy_sim = build_fuzzy_engine()

    # State variables
    last_lat, last_lon, last_time = None, None, None
    recent_jumps = deque(maxlen=8)
    
    print("System Ready. Logging...")

    try:
        while True:
            # -- Read Sensors --
            # Note: We read sensors every loop, even if GPS is slow
            sensor_data = sensors.get_data()
            
            # -- Read GPS --
            try:
                line = ser.readline().decode('latin-1').strip()
            except: 
                line = ""
            
            gps = parse_gga(line)
            
            if not gps["fix"] or gps["sats"] < MIN_SATS:
                time.sleep(0.01)
                continue

            # -- GPS Trust Calculation (Existing Logic) --
            # Calculate Jump
            jump = 0.0
            t_now = time.time()
            if last_lat:
                dt = t_now - last_time
                if dt > 0: jump = haversine_m(last_lat, last_lon, gps['lat'], gps['lon'])
            last_lat, last_lon, last_time = gps['lat'], gps['lon'], t_now
            recent_jumps.append(jump)
            
            # Calculate GPS Score (ANFIS or Fuzzy)
            gps_score = 0.5
            if model:
                # Prepare Inputs: [HDOP, Sats, Jump_Std_Dev]
                vib_hist = np.std(recent_jumps) if len(recent_jumps)>1 else 0
                x = torch.tensor([[gps['hdop'], gps['sats'], vib_hist]]).float()
                with torch.no_grad():
                    gps_score = float(torch.sigmoid(model(x)))
            elif fuzzy_sim:
                try:
                    fuzzy_sim.input['sats'] = gps['sats']
                    fuzzy_sim.input['hdop'] = min(gps['hdop'], 20)
                    fuzzy_sim.compute()
                    gps_score = fuzzy_sim.output['trust']
                except: gps_score = simple_fuzzy_fallback(gps['sats'], gps['hdop'])
            else:
                gps_score = simple_fuzzy_fallback(gps['sats'], gps['hdop'])

            # -- Altitude Fusion (New Logic) --
            final_alt = fuse_altitude(
                gps['alt'], gps_score, 
                sensor_data['baro_alt'], sensor_data['baro_var'], 
                sensor_data['imu_vib']
            )

            # -- Logging --
            row = [
                now_ts(), round(gps['lat'],6), round(gps['lon'],6),
                round(gps['alt'], 2), round(sensor_data['baro_alt'], 2), round(final_alt, 2),
                gps['sats'], gps['hdop'], round(jump,2), round(sensor_data['imu_vib'],3),
                round(gps_score, 2), "HYBRID"
            ]
            
            with open(CSV_LOG, "a", newline="") as f:
                csv.writer(f).writerow(row)
                
            print(f"Alt: {final_alt:.2f}m (G:{gps['alt']:.1f} B:{sensor_data['baro_alt']:.1f}) | Vib: {sensor_data['imu_vib']:.3f}g | GPS Trust: {int(gps_score*100)}%")
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
