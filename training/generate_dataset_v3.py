import csv
import random
import math

# ------------------------------------
# CONFIG
# ------------------------------------
N = 6000
OUT_CSV = "supervisor_log_v3.csv"

# Helper: smooth clamp
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ------------------------------------
# TRUST SCORE RULES (IMPROVED v3)
# ------------------------------------
def compute_trust(hdop, sats, vib):

    # 1) HDOP contribution
    if hdop < 0.8:
        w_hdop = 0.90
    elif hdop < 1.2:
        w_hdop = 0.80
    elif hdop < 2.0:
        w_hdop = 0.65
    elif hdop < 3.0:
        w_hdop = 0.45
    else:
        w_hdop = 0.20

    # 2) Satellites contribution
    if sats >= 20:
        w_sat = 0.95
    elif sats >= 15:
        w_sat = 0.85
    elif sats >= 10:
        w_sat = 0.60
    elif sats >= 7:
        w_sat = 0.40
    else:
        w_sat = 0.20

    # 3) Vibration / position jump contribution
    if vib < 0.2:
        w_vib = 0.90
    elif vib < 0.5:
        w_vib = 0.70
    elif vib < 1.0:
        w_vib = 0.45
    else:
        w_vib = 0.20

    # Combined trust score
    trust = (0.45*w_hdop + 0.35*w_sat + 0.20*w_vib)

    # final cleanup
    trust = clamp(trust, 0.0, 1.0)

    return round(trust, 4)

# ------------------------------------
# MAIN DATA GENERATOR
# ------------------------------------
def generate():
    print("Generating dataset v3...")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hdop", "satellites", "vibration_x", "gps_trust_score", "action_taken", "scenario"])

        for i in range(N):

            # realistic ranges
            hdop = round(random.uniform(0.5, 3.5), 2)
            sats = random.randint(6, 22)
            vib = round(abs(random.gauss(0.3, 0.25)), 3)  # gaussian for realistic jumps

            trust = compute_trust(hdop, sats, vib)

            # Action rules (same as flight logic)
            if trust > 0.85:
                action = "GPS_OK"
            elif trust > 0.55:
                action = "CAUTION"
            else:
                action = "SWITCH_TO_NON_GPS"

            # scenario tags (for monitoring)
            if hdop < 0.8 and sats >= 20:
                sc = "Excellent_GPS"
            elif hdop < 1.5:
                sc = "Good_GPS"
            elif hdop < 2.5:
                sc = "Medium_GPS"
            else:
                sc = "Poor_GPS"

            writer.writerow([hdop, sats, vib, trust, action, sc])

    print(f"Dataset created -> {OUT_CSV}")

# ------------------------------------
if __name__ == "__main__":
    generate()
