import serial
import pynmea2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 9600

# ------------------------------------------------------------
#  PHASE 1: FUZZY LOGIC SETUP (2-input simplified model)
# ------------------------------------------------------------

satellites = ctrl.Antecedent(np.arange(0, 16, 1), 'satellites')
hdop = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'hdop')
gps_trust = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'gps_trust_score')

# Membership functions
satellites['FEW'] = fuzz.trimf(satellites.universe, [0, 0, 6])
satellites['OKAY'] = fuzz.trimf(satellites.universe, [4, 8, 12])
satellites['MANY'] = fuzz.trimf(satellites.universe, [10, 16, 16])

hdop['EXCELLENT'] = fuzz.trimf(hdop.universe, [0, 1, 1.5])
hdop['GOOD'] = fuzz.trimf(hdop.universe, [1.4, 2, 3])
hdop['POOR'] = fuzz.trimf(hdop.universe, [2.5, 10, 10])

gps_trust['LOW'] = fuzz.trimf(gps_trust.universe, [0, 0, 0.5])
gps_trust['MEDIUM'] = fuzz.trimf(gps_trust.universe, [0.4, 0.6, 0.8])
gps_trust['HIGH'] = fuzz.trimf(gps_trust.universe, [0.7, 1, 1])

rule1 = ctrl.Rule(satellites['FEW'] | hdop['POOR'], gps_trust['LOW'])
rule2 = ctrl.Rule(satellites['OKAY'] & hdop['GOOD'], gps_trust['MEDIUM'])
rule3 = ctrl.Rule(satellites['MANY'] & hdop['EXCELLENT'], gps_trust['HIGH'])

gps_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])
gps_simulator = ctrl.ControlSystemSimulation(gps_ctrl_system)


# ------------------------------------------------------------
#  PHASE 2: SAFE HDOP EXTRACTION FUNCTION
# ------------------------------------------------------------

def extract_hdop(msg):
    """Extract HDOP from any GGA message safely."""
    for attr in ["hdop", "horizontal_dil", "horizontal_dilution"]:
        if hasattr(msg, attr):
            value = getattr(msg, attr)
            if value not in [None, "", " "]:
                return float(value)
    return None


# ------------------------------------------------------------
#  PHASE 3: MAIN GPS READING LOOP
# ------------------------------------------------------------

def main():
    print("\n--- Direct GPS Supervisor Initialized ---")
    print(f"Listening for GPS data on {SERIAL_PORT} at {BAUD_RATE} baud...")
    print("Place antenna near a window for best signal.\n")

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    while True:
        try:
            raw_line = ser.readline().decode(errors='ignore').strip()

            if raw_line:
                print("RAW:", raw_line)

            # Only process GGA
            if "GGA" in raw_line:
                try:
                    msg = pynmea2.parse(raw_line)

                    # Extract satellites
                    sats = int(msg.num_sats) if msg.num_sats else None

                    # Extract HDOP safely
                    hdop_val = extract_hdop(msg)

                    if sats is None or hdop_val is None:
                        print("Error: Missing sats or hdop")
                        continue

                    # Run through fuzzy engine
                    gps_simulator.input['satellites'] = sats
                    gps_simulator.input['hdop'] = hdop_val
                    gps_simulator.compute()

                    trust = gps_simulator.output['gps_trust_score']

                    print("\n--- LIVE GPS DATA ---")
                    print(f"  Satellites: {sats}")
                    print(f"  HDOP:       {hdop_val:.2f}")
                    print(f"  Trust Score: {trust:.2f}")

                    if trust >= 0.7:
                        print("  ACTION: GPS Signal is TRUSTED.")
                    else:
                        print("  ACTION: **GPS Signal is UNRELIABLE!**")
                    print("-" * 45)

                except Exception as e:
                    print("Parse error:", e)
                    continue

        except Exception as e:
            print("Error:", e)
            continue


if __name__ == "__main__":
    main()
