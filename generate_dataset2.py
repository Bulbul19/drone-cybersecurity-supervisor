import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
NUM_SAMPLES = 6000
CSV_OUT = "anfis_training_dataset2.csv"

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def clip(x, lo, hi):
    return max(lo, min(hi, x))

# --------------------------------------------------
# MAIN GENERATOR
# --------------------------------------------------
rows = []

def norm(x, maxv):
    return np.clip(x / maxv, 0, 1)

for _ in tqdm(range(NUM_SAMPLES)):

    # Base GPS quality
    sats = np.random.randint(5, 16)
    hdop = np.random.uniform(0.6, 6.0)

    # Motion state
    moving = np.random.rand() < 0.35

    if moving:
        accel_vib = np.random.uniform(0.2, 1.5)
        gyro_mag = np.random.uniform(30, 200)
    else:
        accel_vib = np.random.uniform(0.0, 0.05)
        gyro_mag = np.random.uniform(0.0, 5.0)

    # GPS jump
    if np.random.rand() < 0.25:  # spoof-like
        jump_m = np.random.uniform(15, 80)
    else:
        jump_m = np.random.uniform(0.0, 5.0)

    # Baro-GPS inconsistency
    alt_err = (
        0.4 * jump_m +
        np.random.uniform(0, 5)
    )

    # --- Trust computation ---
    trust = 1.0

    trust -= 0.35 * norm(jump_m, 80)
    trust -= 0.30 * norm(alt_err, 50)
    trust -= 0.20 * norm(hdop, 6)

    # Spoofing condition: big jump, stable IMU
    if jump_m > 15 and accel_vib < 0.05 and gyro_mag < 5:
        trust -= 0.35

    # High motion uncertainty
    if accel_vib > 0.5 or gyro_mag > 80:
        trust -= 0.25

    # Satellite protection
    if sats >= 12:
        trust += 0.10
    elif sats <= 6:
        trust -= 0.10

    # Noise
    trust += np.random.normal(0, 0.03)

    rows.append([
        sats,
        clip(hdop, 0, 20),
        clip(jump_m, 0, 100),
        clip(alt_err, 0, 50),
        clip(accel_vib, 0, 2),
        clip(gyro_mag, 0, 300),
        clip(trust, 0, 1)
    ])

df = pd.DataFrame(rows, columns=[
    "sats",
    "hdop",
    "jump_m",
    "alt_err",
    "accel_vib",
    "gyro_mag",
    "trust"
])

df.to_csv(CSV_OUT, index=False)
print(f"[OK] Dataset saved to {CSV_OUT}")
