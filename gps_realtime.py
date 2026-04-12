import torch
import torch.nn as nn
import serial
import json
import numpy as np
import time
import pynmea2

# -----------------------------
# USER CONFIG
# -----------------------------
MODEL_PATH = "anfis_3in1out.pth"
META_PATH = "anfis_3in1out_meta.json"
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 9600
# -----------------------------

device = torch.device("cpu")

# --------------------------------------------------------------------
# Load model and metadata
# --------------------------------------------------------------------
class ANFISModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load metadata
with open(META_PATH, "r") as f:
    meta = json.load(f)

mu = np.array(meta["mu"])
sigma = np.array(meta["sigma"])

# Initialize model
model = ANFISModel(3, 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model + Metadata Loaded Successfully!")

# --------------------------------------------------------------------
# GPS Reading Helpers
# --------------------------------------------------------------------
prev_lat = None
prev_lon = None
prev_time = None

def compute_velocity(lat, lon, timestamp):
    global prev_lat, prev_lon, prev_time

    if prev_lat is None:
        prev_lat = lat
        prev_lon = lon
        prev_time = timestamp
        return 0.0, 0.0, 0.0

    dt = timestamp - prev_time
    if dt <= 0:
        return 0.0, 0.0, 0.0

    dlat = lat - prev_lat
    dlon = lon - prev_lon
    dtime = dt

    prev_lat = lat
    prev_lon = lon
    prev_time = timestamp

    return dlat, dlon, dtime

def normalize_input(x):
    return (x - mu) / sigma

# --------------------------------------------------------------------
# Start reading GPS
# --------------------------------------------------------------------
print("Opening GPS Port:", SERIAL_PORT)
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

print("Starting Real-Time GPS Processing...")

while True:
    try:
        line = ser.readline().decode("ascii", errors="ignore")

        if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
            msg = pynmea2.parse(line)

            lat = float(msg.latitude)
            lon = float(msg.longitude)
            timestamp = time.time()

            # Compute velocity change
            dlat, dlon, dt = compute_velocity(lat, lon, timestamp)

            # Prepare model input (3 inputs)
            X = np.array([dlat, dlon, dt], dtype=np.float32)
            X_norm = normalize_input(X)

            X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)
            output = model(X_t).item()

            gps_trust_score = max(0.0, min(1.0, output))  # Clamp 0–1

            print(f"Lat: {lat:.6f}, Lon: {lon:.6f} | Trust Score = {gps_trust_score:.3f}")

    except Exception as e:
        print("Error:", e)

