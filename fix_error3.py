import smbus2
import time
import math

# ======================
# I2C SETUP
# ======================
bus = smbus2.SMBus(1)

MPU_ADDR = 0x68
BME_ADDR = 0x76

# ======================
# MPU6050 INIT
# ======================
def init_mpu():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0x00)  # Wake up
    bus.write_byte_data(MPU_ADDR, 0x1C, 0x00)  # ±2g
    bus.write_byte_data(MPU_ADDR, 0x1B, 0x00)  # ±250°/s

def read_word(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    val = (high << 8) | low
    return val - 65536 if val > 32767 else val

def read_imu():
    ax = read_word(MPU_ADDR, 0x3B) / 16384.0
    ay = read_word(MPU_ADDR, 0x3D) / 16384.0
    az = read_word(MPU_ADDR, 0x3F) / 16384.0

    gx = read_word(MPU_ADDR, 0x43) / 131.0
    gy = read_word(MPU_ADDR, 0x45) / 131.0
    gz = read_word(MPU_ADDR, 0x47) / 131.0

    accel_vib = math.sqrt(ax*ax + ay*ay + az*az)
    gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)

    return accel_vib, gyro_mag

# ======================
# BME280 INIT (SAFE MODE)
# ======================
def init_bme():
    bus.write_byte_data(BME_ADDR, 0xF4, 0x27)  # temp+press normal mode
    bus.write_byte_data(BME_ADDR, 0xF5, 0xA0)

def read_bme_pressure():
    msb = bus.read_byte_data(BME_ADDR, 0xF7)
    lsb = bus.read_byte_data(BME_ADDR, 0xF8)
    xlsb = bus.read_byte_data(BME_ADDR, 0xF9)

    adc_p = (msb << 12) | (lsb << 4) | (xlsb >> 4)
    return adc_p

# ======================
# CONFIDENCE CALCULATION
# ======================
def confidence_scores(accel_vib, gyro_mag):
    acc_low = max(0, min(1, (0.05 - accel_vib) / 0.05))
    acc_high = max(0, min(1, (accel_vib - 0.3) / 1.7))

    gyro_low = max(0, min(1, (10 - gyro_mag) / 10))
    gyro_high = max(0, min(1, (gyro_mag - 80) / 420))

    return acc_low, acc_high, gyro_low, gyro_high

# ======================
# MAIN LOOP
# ======================
init_mpu()
init_bme()

print("IMU + Barometer running...\n")

while True:
    accel_vib, gyro_mag = read_imu()
    pressure = read_bme_pressure()

    acc_low, acc_high, gyro_low, gyro_high = confidence_scores(accel_vib, gyro_mag)

    print(f"Accel Vib: {accel_vib:.3f} g | Gyro Mag: {gyro_mag:.1f} dps")
    print(f"Pressure RAW: {pressure}")
    print(f"AccLow:{acc_low:.2f} AccHigh:{acc_high:.2f} GyLow:{gyro_low:.2f} GyHigh:{gyro_high:.2f}")
    print("-" * 60)

    time.sleep(1)
