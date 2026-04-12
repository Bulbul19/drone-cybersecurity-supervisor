import time
import serial
import pynmea2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --------------------------------------------------------------------------
# PHASE 1: SETUP THE FUZZY LOGIC ENGINE (Simplified 2-Input Version)
# --------------------------------------------------------------------------
# Define the input and output variables
satellites = ctrl.Antecedent(np.arange(0, 16, 1), 'satellites')
hdop = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'hdop')
gps_trust = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'gps_trust_score')

# Define the membership functions
satellites['FEW'] = fuzz.trimf(satellites.universe, [0, 0, 6])
satellites['OKAY'] = fuzz.trimf(satellites.universe, [4, 8, 12])
satellites['MANY'] = fuzz.trimf(satellites.universe, [10, 16, 16])

hdop['EXCELLENT'] = fuzz.trimf(hdop.universe, [0, 1, 1.5])
hdop['GOOD'] = fuzz.trimf(hdop.universe, [1.4, 2, 3])
hdop['POOR'] = fuzz.trimf(hdop.universe, [2.5, 10, 10])

gps_trust['LOW'] = fuzz.trimf(gps_trust.universe, [0, 0, 0.5])
gps_trust['MEDIUM'] = fuzz.trimf(gps_trust.universe, [0.4, 0.6, 0.8])
gps_trust['HIGH'] = fuzz.trimf(gps_trust.universe, [0.7, 1, 1])

# Define the simplified Rulebook
rule1 = ctrl.Rule(satellites['FEW'] | hdop['POOR'], gps_trust['LOW'])
rule2 = ctrl.Rule(satellites['OKAY'] & hdop['GOOD'], gps_trust['MEDIUM'])
rule3 = ctrl.Rule(satellites['MANY'] & hdop['EXCELLENT'], gps_trust['HIGH'])

# Create the Control System
gps_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])
gps_simulator = ctrl.ControlSystemSimulation(gps_ctrl_system)

# --------------------------------------------------------------------------
# PHASE 2: LIVE GPS CONNECTION & SUPERVISOR LOOP
# --------------------------------------------------------------------------
# This is the serial port on the Raspberry Pi's GPIO pins
SERIAL_PORT = '/dev/serial0' 
# This is the speed. 9600 is the most common default.
BAUD_RATE = 9600 

def main():
    """Main loop to read GPS data and run the fuzzy supervisor."""
    print(f"--- Direct GPS Supervisor Initialized (2-Input Version) ---")
    print(f"Listening for GPS data on {SERIAL_PORT} at {BAUD_RATE} baud...")
    print("Place your antenna near a window or outdoors to get a fix.")

    try:
        # Initialize the serial connection
          ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)

        while True:
          try:
            raw_line = ser.readline().decode(errors='ignore').strip()

            # Only process if something was read
            if raw_line:
                pass
                # print("RAW:", raw_line)  # optional debug

            # Only process GGA messages
            if "GGA" in raw_line:
                try:
                    msg = pynmea2.parse(raw_line)

                    # Check for valid GPS fix
                    if msg.gps_qual in [1, 2]:
                        sats = int(msg.num_sats)
                        hdop_val = extract_hdop(msg)

                        current_lat = msg.latitude
                        current_lon = msg.longitude

                        # Distance calculation
                        current_distance = 0.0
                        if LAST_LAT is not None and LAST_LON is not None:
                            distance_m = haversine_distance(
                                LAST_LAT, LAST_LON, current_lat, current_lon
                            )
                            current_distance = min(distance_m, 100.0)

                        LAST_LAT = current_lat
                        LAST_LON = current_lon

                        if sats is None or hdop_val is None:
                            print("Error: Missing sats or hdop")
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

                        if trust >= 0.5:
                            print("  ACTION: GPS Signal is TRUSTED.")
                        else:
                            print("  ACTION: **GPS Signal is UNRELIABLE (Possible Spoofing)!**")

                        print("-" * 45)

                except Exception:
                    continue

        except Exception:
            continue
 try:
                # Read one line of NMEA text from the GPS module
                line = ser.readline().decode('ascii', errors='replace')
                
                # Check if the line is a GPGGA or GNGGA sentence
                if 'GGA' in line:
                    msg = pynmea2.parse(line)
                    
                    # Check that the message has valid data
                    if msg.num_sats and msg.horiz_dil:
                        sats = int(msg.num_sats)
                        current_hdop = float(msg.horiz_dil)
                        
                        # Feed the live evidence into the fuzzy logic simulator
                        gps_simulator.input['satellites'] = sats
                        gps_simulator.input['hdop'] = current_hdop

                        # Run the engine to compute the trust score
                        gps_simulator.compute()
                        trust_score = gps_simulator.output['gps_trust_score']

                        # Print the results
                        print(f"--- LIVE GPS DATA ---")
                        print(f"  Inputs: Sats={sats}, HDOP={current_hdop:.2f}")
                        print(f"  > GPS Trust Score: {trust_score:.2f}")

                        if trust_score >= 0.5:
                            print("  ACTION: GPS Signal is TRUSTED.")
                        else:
                            print("  ACTION: **GPS Signal is UNRELIABLE!**")

                        print("-" * 45)
                    
            except (pynmea2.ParseError, ValueError, AttributeError) as e:
                # Ignore corrupted lines or messages without the data we need
                pass
                
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n--- Supervisor shutting down ---")
    except serial.SerialException as e:
        print(f"\nSerial Error: {e}")
        print("Could not open serial port. Please check hardware connections and permissions.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
