#!/usr/bin/env python3
import serial
import time
import csv
import numpy as np
from datetime import datetime

# -------------------------------------------------------------
# GPS SERIAL
# -------------------------------------------------------------
gps_serial = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

def read_gps_line():
    """Reads one raw NMEA line from GPS module."""
    try:
        return gps_serial.readline().decode(errors="ignore").strip()
    except:
        return ""


def parse_nmea(line):
    """
    Parses GGA string and returns dict:
    {lat, lon, sats, hdop, valid_fix}
    If invalid or sats==0, valid_fix=False
    """
    if not line.startswith("$GNGGA") and not line.startswith("$GPGGA"):
        return {"valid_fix": False}

    parts = line.split(",")
    try:
        sats = int(parts[7])
        hdop = float(parts[8])
        if sats == 0:
            return {"valid_fix": False}

        raw_lat = parts[2]
        raw_lon = parts[4]

        lat = float(raw_lat[:2]) + float(raw_lat[2:]) / 60
        lon = float(raw_lon[:3]) + float(raw_lon[3:]) / 60

        return {"lat": lat, "lon": lon, "sats": sats, "hdop": hdop, "valid_fix": True}
    except:
        return {"valid_fix": False}


# -------------------------------------------------------------
# MODEL PLACEHOLDERS (replace with your real models)
# -------------------------------------------------------------
def anfis_predict(lat, lon, sats, hdop, jump, vibration):
    """Return ANFIS prediction 0..1"""
    return np.random.random()  # placeholder


def fuzzy_fallback_predict(lat, lon, sats, hdop, jump, vibration):
    """Return fuzzy fallback score 0..1"""
    return np.random.random()  # placeholder


# -------------------------------------------------------------
# HELPERS FOR FEATURES
# -------------------------------------------------------------
last_lat = None
last_lon = None
last_vib = 0

def compute_jump(lat, lon):
    global last_lat, last_lon
    if last_lat is None or last_lon is None:
        last_lat, last_lon = lat, lon
        return 0.0
    d = np.sqrt((lat - last_lat) ** 2 + (lon - last_lon) ** 2) * 111000
    last_lat, last_lon = lat, lon
    return d


def compute_vibration(jump):
    global last_vib
    last_vib = 0.9 * last_vib + 0.1 * jump
    return last_vib


def classify(score):
    if score > 0.67:
        return "GOOD"
    elif score > 0.34:
        return "JAMMED/WEAK"
    else:
        return "SPOOFED"


# -------------------------------------------------------------
# CSV LOGGER
# -------------------------------------------------------------
csv_file = "gps_log.csv"
csv_header = ["timestamp","lat","lon","sats","hdop","jump","vibration","anfis","fuzzy","final","label"]

first_write = not (csv_file in globals())

def log_csv(row):
    newfile = False
    try:
        with open(csv_file, "x") as f:
            newfile = True
    except:
        pass
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if newfile:
            writer.writerow(csv_header)
        writer.writerow(row)


# -------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------
def main():
    print("Starting pipeline: GPS → ANFIS → fuzzy fallback → CSV logger")
    print("[INFO] Listening for NMEA on /dev/serial0 @ 9600 ... Press Ctrl-C to stop.")

    while True:
        line = read_gps_line()
        parsed = parse_nmea(line)
        if not parsed["valid_fix"]:
            continue  # SILENT: no NO-FIX printed

        lat = parsed["lat"]
        lon = parsed["lon"]
        sats = parsed["sats"]
        hdop = parsed["hdop"]

        # Block spoof values like sats=0 automatically
        if sats < 3 or hdop >= 10:
            continue  # silent drop of garbage

        jump = compute_jump(lat, lon)
        vib = compute_vibration(jump)

        anfis = anfis_predict(lat, lon, sats, hdop, jump, vib)
        if anfis > 0.05:
            final_score = anfis
            source = "ANFIS"
        else:
            fuzzy = fuzzy_fallback_predict(lat, lon, sats, hdop, jump, vib)
            final_score = fuzzy
            source = "FUZZY_FALLBACK"

        label = classify(final_score)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_csv([timestamp, lat, lon, sats, hdop, jump, vib, anfis, fuzzy if source!="ANFIS" else 0, final_score, label])

        print(f"[{timestamp}] Lat:{lat:.6f} Lon:{lon:.6f} sats:{sats} hdop:{hdop:.2f} "
              f"jump:{jump:.2f}m vib:{vib:.2f} final:{final_score:.3f} => {label} ({source})")

        time.sleep(0.2)


# -------------------------------------------------------------
# START
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped manually.")
