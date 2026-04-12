import serial
import pynmea2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from math import radians, sin, cos, sqrt, atan2

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 9600

LAST_LAT = None
LAST_LON = None

# ------------------------------------------------------------
#  FUZZY LOGIC SETUP
# ------------------------------------------------------------

satellites = ctrl.Antecedent(np.arange(0, 16, 1), 'satellites')
hdop = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'hdop')
distance_jump = ctrl.Antecedent(np.arange(0, 101, 1), 'distance_jump')
gps_trust = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'gps_trust_score')

satellites['FEW'] = fuzz.trimf(satellites.universe, [0, 0, 6])
satellites['OKAY'] = fuzz.trimf(satellites.universe, [4, 8, 12])
satellites['MANY'] = fuzz.trimf(satellites.universe, [10, 16, 16])

hdop['EXCELLENT'] = fuzz.trimf(hdop.universe, [0, 1, 1.5])
hdop['GOOD'] = fuzz.trimf(hdop.universe, [1.4, 2, 3])
hdop['POOR'] = fuzz.trimf(hdop.universe, [2.5, 10, 10])

distance_jump['NORMAL'] = fuzz.trimf(distance_jump.universe, [0, 5, 15])
distance_jump['SUSPICIOUS'] = fuzz.trimf(distance_jump.universe, [10, 40, 70])
distance_jump['IMPOSSIBLE'] = fuzz.trimf(distance_jump.universe, [60, 100, 100])

gps_trust['LOW'] = fuzz.trimf(gps_trust.universe, [0, 0, 0.5])
gps_trust['MEDIUM'] = fuzz.trimf(gps_trust.universe, [0.4, 0.6, 0.8])
gps_trust['HIGH'] = fuzz.trimf(gps_trust.universe, [0.7, 1, 1])

rule1 = ctrl.Rule(satellites['FEW'] | hdop['POOR'], gps_trust['LOW'])
rule2 = ctrl.Rule(distance_jump['IMPOSSIBLE'], gps_trust['LOW'])
rule3 = ctrl.Rule(distance_jump['SUSPICIOUS'] & satellites['OKAY'], gps_trust['MEDIUM'])
rule4 = ctrl.Rule(satellites['OKAY'] & hdop['GOOD'] & distance_jump['NORMAL'], gps_trust['MEDIUM'])
rule5 = ctrl.Rule(satellites['MANY'] & hdop['EXCELLENT'] & distance_jump['NORMAL'], gps_trust['HIGH'])

gps_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
gps_simulator = ctrl.ControlSystemSimulation(gps_ctrl_system)

# ------------------------------------------------------------
#  SAFE HDOP EXTRACTION + HAVERSINE
# ------------------------------------------------------------

def extract_hdop(msg):
    for attr in ["hdop", "horizontal_dil", "horizontal_dilution"]:
        if hasattr(msg, attr):
            val = getattr(msg, attr)
            if val not in [None, "", " "]:
                return float(val)
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ------------------------------------------------------------
#  MAIN LOOP
# ------------------------------------------------------------

def main():
    global LAST_LAT, LAST_LON

    print("\n--- Direct GPS Supervisor Initialized (3-Input) ---")
    print(f"Listening for GPS on {SERIAL_PORT} @ {BAUD_RATE}\n")

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    while True:
        try:
            raw_line = ser.readline().decode(errors='ignore').strip()

            if not raw_line:
                continue   # <-- FIXED: ensures safe indentation

            if "GGA" in raw_line:
                try:
                    msg = pynmea2.parse(raw_line)

                    if msg.gps_qual in [1, 2]:
                        sats = int(msg.num_sats)
                        hdop_val = extract_hdop(msg)

                        current_lat = msg.latitude
                        current_lon = msg.longitude

                        # Distance jump calculation
                        current_distance = 0.0
                        if LAST_LAT is not None and LAST_LON is not None:
                            jump = haversine_distance(LAST_LAT, LAST_LON, current_lat, current_lon)
                            current_distance = min(jump, 100)

                        LAST_LAT, LAST_LON = current_lat, current_lon

                        if sats is None or hdop_val is None:
                            print("Missing sats or hdop")
                            continue

                        gps_simulator.input['satellites'] = sats
                        gps_simulator.input['hdop'] = hdop_val
                        gps_simulator.input['distance_jump'] = current_distance
                        gps_simulator.compute()

                        trust = gps_simulator.output['gps_trust_score']

                        print("\n--- LIVE GPS DATA ---")
                        print(f"  Satellites: {sats}")
                        print(f"  HDOP: {hdop_val:.2f}")
                        print(f"  Distance Jump: {current_distance:.2f} m")
                        print(f"  Trust Score: {trust:.2f}")

                        if trust >= 0.7:
                            print("  ACTION: GPS TRUSTED")
                        else:
                            print("  ACTION: **GPS UNRELIABLE — SPOOFING RISK**")

                        print("--------------------------------------------")

                except Exception:
                    continue

        except Exception:
            continue


if __name__ == "__main__":
    main()
