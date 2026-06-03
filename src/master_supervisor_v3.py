#!/usr/bin/env python3
"""
master_supervisor_v3.py

Updates:
- Shows "Waiting for GPS" status so the script doesn't look dead.
- Handles Sensor Calibration.
- Fuses Barometer + GPS + IMU.
"""

import time
import math
import csv
import json
import os
from pathlib import Path
from collections import deque
from datetime import datetime

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
    print("Error: RPi.bme280 not installed.")
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
CSV_LOG = "realtime_log_v3.csv"
BACKUP_DIR = "backups"
I2C_BUS = 1
MPU6050_ADDR = 0x68
BME280_ADDR = 0x76
MIN_SATS = 3

# -----------------------------
# SENSOR MANAGER
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
            # Wake up & Config
            self.bus.write_byte_data(self.mpu_addr, 0x6B, 0)
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
        total = 0
        samples = 50
        for _ in range(samples):
            data = self.read_raw_imu()
            total += data['total_g']
            time.sleep(0.01)
        self.resting_g = total / samples
        print(f"[SENSORS] Calibration Complete. Resting G: {self.resting_g:.3f}")

    def set_ground_pressure(self):
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
        if not self.imu_available:
            return {'total_g': 1.0, 'gyro_mag': 0.0}
        try:
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
            return {'total_g': total_g, 'gyro_mag': gyro_mag}
        except:
             return {'total_g': 1.0, 'gyro_mag': 0.0}

    def get_data(self):
        raw_imu = self.read_raw_imu()
        vibration = abs(self.resting_g - raw_imu['total_g'])
        
        alt = 0.0
        velocity_var = 0.0
        
        if self.baro_available:
            try:
                d = bme280.sample(self.bus, self.baro_addr, self.cal_params)
                alt = 44330 * (1.0 - pow(d.pressure / self.ground_pressure, 0.1903))
                t_now = time.time()
                dt = t_now - self.last_baro_time
                if dt > 0: velocity_var = abs(alt - self.last_baro_alt) / dt
                self.last_baro_alt = alt
                self.last_baro_time = t_now
            except: pass

        return {
            'imu_vib': vibration, 'baro_alt': alt, 'baro_var': velocity_var
        }

# -----------------------------
# LOGGING & UTILS
# -----------------------------
def now_ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ensure_dir(path): os.makedirs(path, exist_ok=True)
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_gga(line):
    if not line or "$G" not in line: return {"fix": False, "sats": 0}
    try:
        if pynmea2:
            msg = pynmea2.parse(line.strip())
            if isinstance(msg, pynmea2.types.talker.GGA) and msg.gps_qual > 0:
                return {
                    "fix": True, "lat": msg.latitude, "lon": msg.longitude,
                    "sats": int(msg.num_sats), "hdop": float(msg.horizontal_dil),
                    "alt": float(msg.altitude)
                }
            else:
                return {"fix": False, "sats": int(msg.num_sats) if hasattr(msg, 'num_sats') else 0}
    except: pass
    return {"fix": False, "sats": 0}

# -----------------------------
# FUSION
# -----------------------------
def fuse_altitude(gps_alt, gps_trust, baro_alt, baro_var, imu_vib):
    baro_trust = 1.0
    if baro_var > 2.0: baro_trust = 0.5
    if imu_vib > 0.3: baro_trust *= 0.5
    total = gps_trust + baro_trust
    if total == 0: return baro_alt
    return (gps_alt * gps_trust + baro_alt * baro_trust) / total

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("--- Supervisor v3: Starting ---")
    ensure_dir(BACKUP_DIR)
    
    # 1. Hardware Init
    sensors = SensorManager(bus_number=I2C_BUS)
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except:
        print("Serial Error")
        return

    # 2. Setup CSV
    if not Path(CSV_LOG).exists():
        with open(CSV_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","lat","lon","gps_alt","baro_alt","fused_alt","sats","vib","trust"])

    # State
    last_lat, last_lon, last_time = None, None, None
    last_log_time = 0
    
    print("System Ready. Waiting for GPS fix...")

    try:
        while True:
            # A. Read Sensors (Always)
            s_data = sensors.get_data()
            
            # B. Read GPS
            try:
                line = ser.readline().decode('latin-1').strip()
            except: 
                line = ""
            
            gps = parse_gga(line)
            
            # C. Check GPS Status
            if not gps["fix"] or gps["sats"] < MIN_SATS:
                # FIX: Print status every 1 second so you know it's alive
                if time.time() - last_log_time > 1.0:
                    print(f"[WAITING GPS] Sats: {gps.get('sats', 0)} | Baro: {s_data['baro_alt']:.2f}m | Vib: {s_data['imu_vib']:.3f}g")
                    last_log_time = time.time()
                time.sleep(0.01)
                continue

            # D. If Fixed - Run Fusion
            jump = 0.0
            t_now = time.time()
            if last_lat:
                dt = t_now - last_time
                if dt > 0: jump = haversine_m(last_lat, last_lon, gps['lat'], gps['lon'])
            last_lat, last_lon, last_time = gps['lat'], gps['lon'], t_now
            
            # Simple Trust Score (0.0 to 1.0)
            gps_trust = 0.5
            if gps['hdop'] < 2.0: gps_trust += 0.3
            if gps['sats'] > 8: gps_trust += 0.2

            # Fuse Altitude
            final_alt = fuse_altitude(
                gps['alt'], gps_trust, 
                s_data['baro_alt'], s_data['baro_var'], 
                s_data['imu_vib']
            )

            # E. Log & Print
            with open(CSV_LOG, "a", newline="") as f:
                csv.writer(f).writerow([
                    now_ts(), round(gps['lat'],6), round(gps['lon'],6),
                    round(gps['alt'], 2), round(s_data['baro_alt'], 2), round(final_alt, 2),
                    gps['sats'], round(s_data['imu_vib'],3), round(gps_trust, 2)
                ])
                
            print(f"FIXED | Alt: {final_alt:.2f}m (G:{gps['alt']:.1f} B:{s_data['baro_alt']:.1f}) | Vib: {s_data['imu_vib']:.3f}g")
            last_log_time = time.time()

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
