import time
import random # To simulate GPS for this demo
from sensors import SensorManager
from fuzzy_fusion import FuzzyTrustSystem

def main():
    print("--- Drone Trust Monitor Starting ---")
    
    sensors = SensorManager()
    fuzzy = FuzzyTrustSystem()
    
    # Fake GPS data generator (Replace with real GPS read)
    sim_gps_alt = 0.0
    
    try:
        while True:
            # --- 1. Gather Data ---
            imu_data = sensors.get_imu_data()
            baro_data = sensors.get_baro_data()
            
            # Simulate GPS (Drifting slightly)
            sim_gps_alt += random.uniform(-0.1, 0.1) 
            gps_sats = 14
            gps_hdop = 0.8
            
            # --- 2. Calculate Trust Scores ---
            
            # IMU Trust: Checks for vibration/crash
            trust_imu = fuzzy.evaluate_imu_trust(
                imu_data['vibration'], 
                imu_data['gyro_magnitude']
            )
            
            # Baro Trust: Checks for impossible spikes
            trust_baro = fuzzy.evaluate_baro_trust(
                baro_data['velocity_variance']
            )
            
            # GPS Trust: Checks signal health
            trust_gps = fuzzy.evaluate_gps_trust(gps_sats, gps_hdop)
            
            # --- 3. Fuse Data ---
            # If IMU trust is low (shaking), we might rely less on GPS position updates 
            # (since we can't verify them with Inertial guidance)
            
            final_alt = fuzzy.fuse_altitude(
                baro_data['altitude'], trust_baro,
                sim_gps_alt, trust_gps
            )
            
            # --- 4. System Health Report ---
            print("\n--- SENSOR FUSION REPORT ---")
            print(f"IMU VIB: {imu_data['vibration']:.3f}g  | Trust: {int(trust_imu*100)}%")
            print(f"BARO VAR: {baro_data['velocity_variance']:.2f} | Trust: {int(trust_baro*100)}%")
            print(f"GPS HDOP: {gps_hdop}       | Trust: {int(trust_gps*100)}%")
            print("-" * 30)
            print(f"Baro Alt: {baro_data['altitude']:.2f}m")
            print(f"GPS Alt:  {sim_gps_alt:.2f}m")
            print(f"FUSED ALTITUDE: {final_alt:.2f}m")
            
            # Physical Warning
            if trust_imu < 0.5:
                print(">> WARNING: HIGH VIBRATION DETECTED <<")
            
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nMonitor Stopped.")

if __name__ == "__main__":
    import time
    main()

