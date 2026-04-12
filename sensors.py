import smbus2
import bme280
import math
import time

class SensorManager:
    def __init__(self, bus_number=1):
        """
        Initializes the I2C connection for BME280 (Barometer) and MPU6050 (IMU).
        Includes Auto-Calibration.
        """
        try:
            self.bus = smbus2.SMBus(bus_number)
        except Exception as e:
            print(f"[SENSORS] Fatal Error: Could not open I2C bus {bus_number}: {e}")
            return

        # --- BME280 Setup ---
        self.baro_addr = 0x76 
        self.ground_pressure = 1013.25
        self.last_baro_time = time.time()
        self.last_baro_alt = 0.0
        self.cal_params = None
        
        try:
            self.cal_params = bme280.load_calibration_params(self.bus, self.baro_addr)
            print(f"[SENSORS] Barometer Connected at {hex(self.baro_addr)}")
        except Exception as e:
            print(f"[SENSORS] Barometer Error: {e}")

        # --- MPU6050 Setup ---
        self.mpu_addr = 0x68
        self.resting_g = 1.0 # Default fallback
        
        try:
            # 1. Wake up MPU6050
            self.bus.write_byte_data(self.mpu_addr, 0x6B, 0)
            
            # 2. Set Range to +/- 2g (Register 0x1C = 0)
            self.bus.write_byte_data(self.mpu_addr, 0x1C, 0)
            
            # 3. Set Gyro to +/- 250dps (Register 0x1B = 0)
            self.bus.write_byte_data(self.mpu_addr, 0x1B, 0)
            
            print(f"[SENSORS] MPU6050 Configured at {hex(self.mpu_addr)}")
            
            # 4. Auto-Calibrate (Tare)
            print("[SENSORS] Calibrating IMU (Keep drone still)...")
            total_mag = 0
            samples = 50
            for _ in range(samples):
                data = self._read_raw_imu()
                total_mag += data['total_g']
                time.sleep(0.02)
            
            self.resting_g = total_mag / samples
            print(f"[SENSORS] Calibration Complete. Resting Gravity: {self.resting_g:.3f}g")
            
            # 5. Set Barometer Ground Level
            if self.cal_params:
                press_sum = 0
                for _ in range(10):
                    press_sum += self._read_raw_pressure()
                    time.sleep(0.05)
                self.ground_pressure = press_sum / 10
                print(f"[SENSORS] Ground Pressure Set: {self.ground_pressure:.2f} hPa")
                
        except Exception as e:
            print(f"[SENSORS] MPU6050 Error: {e}")

    def _read_raw_pressure(self):
        if not self.cal_params: return 1013.25
        try:
            data = bme280.sample(self.bus, self.baro_addr, self.cal_params)
            return data.pressure
        except:
            return 1013.25

    def _read_raw_imu(self):
        """Reads raw bytes and converts to G-force/DPS"""
        try:
            raw_data = self.bus.read_i2c_block_data(self.mpu_addr, 0x3B, 14)
            
            def to_signed16(high, low):
                val = (high << 8) | low
                return val - 65536 if val > 32768 else val

            ax = to_signed16(raw_data[0], raw_data[1]) / 16384.0
            ay = to_signed16(raw_data[2], raw_data[3]) / 16384.0
            az = to_signed16(raw_data[4], raw_data[5]) / 16384.0
            
            gx = to_signed16(raw_data[8], raw_data[9]) / 131.0
            gy = to_signed16(raw_data[10], raw_data[11]) / 131.0
            gz = to_signed16(raw_data[12], raw_data[13]) / 131.0

            total_g = math.sqrt(ax*ax + ay*ay + az*az)
            gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)

            return {"total_g": total_g, "gyro_mag": gyro_mag}
        except:
            return {"total_g": 1.0, "gyro_mag": 0.0}

    def get_baro_data(self):
        current_pressure = self._read_raw_pressure()
        altitude = 44330 * (1.0 - pow(current_pressure / self.ground_pressure, 0.1903))
        
        current_time = time.time()
        dt = current_time - self.last_baro_time
        if dt <= 0: dt = 0.01
        velocity = abs(altitude - self.last_baro_alt) / dt
        
        self.last_baro_alt = altitude
        self.last_baro_time = current_time
        
        return {"altitude": altitude, "velocity_variance": velocity}

    def get_imu_data(self):
        data = self._read_raw_imu()
        
        # Calculate Vibration relative to the calibrated Resting Gravity
        # If resting is 0.5g and current is 0.5g, vibration is 0.0
        vibration = abs(self.resting_g - data['total_g'])
        
        return {
            "vibration": vibration,
            "gyro_magnitude": data['gyro_mag']
        }

if __name__ == "__main__":
    manager = SensorManager()
    while True:
        try:
            imu = manager.get_imu_data()
            baro = manager.get_baro_data()
            print(f"Alt: {baro['altitude']:.2f}m | Vib: {imu['vibration']:.3f}g")
            time.sleep(0.5)
        except KeyboardInterrupt:
            break
