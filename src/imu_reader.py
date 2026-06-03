#!/usr/bin/env python3
import time
from smbus2 import SMBus

# MPU-6050 Register Map Constants
# ----------------------------------------------------
# I2C address of the MPU-6050 (0x68 or 0x69, check i2cdetect)
ADDRESS = 0x68
# Power Management Register 1 - wakes up the device
PWR_MGMT_1 = 0x6B
# Accelerometer registers (High and Low bytes for X, Y, Z)
ACCEL_XOUT_H = 0x3B
# Scale factor for 2g range (32768 / 2 = 16384)
ACCEL_SCALE_FACTOR = 16384.0 

# I2C Bus: 1 for Raspberry Pi 2/3/4; 0 for older Pi 1
BUS_NUMBER = 1 

def read_word(reg):
    """Reads two bytes (High byte, Low byte) and combines them into an integer."""
    high = bus.read_byte_data(ADDRESS, reg)
    low = bus.read_byte_data(ADDRESS, reg+1)
    val = (high << 8) + low
    # Convert from 2's complement to signed integer (16-bit)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

def read_accel_data():
    """Reads raw acceleration values for X, Y, and Z axes."""
    accel_x = read_word(ACCEL_XOUT_H)
    accel_y = read_word(ACCEL_XOUT_H + 2) # X registers are separated by 2 bytes
    accel_z = read_word(ACCEL_XOUT_H + 4)
    
    # Convert raw values to Gs (Gravity units)
    ax = accel_x / ACCEL_SCALE_FACTOR
    ay = accel_y / ACCEL_SCALE_FACTOR
    az = accel_z / ACCEL_SCALE_FACTOR
    
    return ax, ay, az

# Initialize I2C bus and MPU-6050
try:
    bus = SMBus(BUS_NUMBER)
    # Wake up the MPU-6050 (write 0 to PWR_MGMT_1 register)
    bus.write_byte_data(ADDRESS, PWR_MGMT_1, 0)
    print(f"[INFO] MPU-6050 initialized on I2C address {hex(ADDRESS)}.")
    
except FileNotFoundError:
    print("[ERROR] I2C bus not found. Check if I2C is enabled in raspi-config.")
    exit()
except Exception as e:
    print(f"[ERROR] Could not communicate with MPU-6050 at {hex(ADDRESS)}: {e}")
    print("Check your wiring and run 'i2cdetect -y 1' to confirm the device address.")
    exit()

# Main reading loop
print("Starting IMU data stream...")
try:
    while True:
        ax, ay, az = read_accel_data()
        
        # Calculate Acceleration Magnitude (a_mag) - useful for feature engineering
        a_mag = (ax**2 + ay**2 + az**2)**0.5
        
        # Print results
        print(f"Accel: X={ax:.3f}g | Y={ay:.3f}g | Z={az:.3f}g | Mag={a_mag:.3f}g")
        
        # Read the IMU data much faster than the 1Hz GPS rate
        time.sleep(0.01) # Approx 100Hz rate

except KeyboardInterrupt:
    print("\nIMU reader stopped by user.")
finally:
    try:
        bus.close()
    except Exception:
        pass
