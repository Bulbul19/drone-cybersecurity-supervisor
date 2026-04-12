import time
import math
import smbus2

# =========================
# MPU6050 SETUP
# =========================
MPU_ADDR = 0x68
bus = smbus2.SMBus(1)

def init_mpu():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)

def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)
    value = (high << 8) | low
    return value - 65536 if value > 32768 else value

def read_mpu():
    ax = read_word(0x3B) / 16384.0
    ay = read_word(0x3D) / 16384.0
    az = read_word(0x3F) / 16384.0

    gx = read_word(0x43) / 131.0
    gy = read_word(0x45) / 131.0
    gz = read_word(0x47) / 131.0

    accel_vib = math.sqrt(ax*ax + ay*ay + az*az)
    gyro_mag = abs(gx) + abs(gy) + abs(gz)

    return accel_vib, gyro_mag


# =========================
# BME280 SETUP
# =========================
i2c = busio.I2C(board.SCL, board.SDA)
bme = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)
bme.sea_level_pressure = 1013.25
REG_CTRL_HUM = 0xF2
REG_CTRL_MEAS = 0xF4
REG_CONFIG = 0xF5
REG_DATA = 0xF7

def init_bme280():
    bus.write_byte_data(BME_ADDR, REG_CTRL_HUM, 0x01)
    bus.write_byte_data(BME_ADDR, REG_CTRL_MEAS, 0x27)
    bus.write_byte_data(BME_ADDR, REG_CONFIG, 0xA0)

def read_bme280_altitude(sea_level_hpa=1013.25):
    data = bus.read_i2c_block_data(BME_ADDR, REG_DATA, 8)
    adc_p = (data[0]<<12) | (data[1]<<4) | (data[2]>>4)
    pressure = adc_p / 25600.0   # simplified
    altitude = 44330.0 * (1.0 - (pressure / sea_level_hpa) ** 0.1903)
    return altitude

# =========================
# MAIN LOOP
# =========================
init_mpu()

while True:
    accel_vib, gyro_mag = read_mpu()
    altitude = bme.altitude

    # =========================
    # FUZZY MEMBERSHIP VALUES
    # =========================
    acc_low = max(0, min(1, (0.05 - accel_vib) / 0.05))
    acc_high = max(0, min(1, (accel_vib - 0.3) / 1.7))

    gyro_low = max(0, min(1, (10 - gyro_mag) / 10))
    gyro_high = max(0, min(1, (gyro_mag - 80) / 420))

    # =========================
    # PRINT OUTPUT
    # =========================
    print(f"Accel Vib   : {accel_vib:.3f} g")
    print(f"Gyro Mag    : {gyro_mag:.2f} deg/s")
    print(f"Altitude    : {altitude:.2f} m")
    print(f"acc_low     : {acc_low:.2f}")
    print(f"acc_high    : {acc_high:.2f}")
    print(f"gyro_low    : {gyro_low:.2f}")
    print(f"gyro_high   : {gyro_high:.2f}")
    print("-" * 40)

    time.sleep(1)

