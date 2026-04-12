import time
import math
import serial
import pynmea2
import smbus2
import board
import busio
import adafruit_bme280.basic as adafruit_bme280

print("\n=== SENSOR FUSION STARTED ===\n")

# ==============================
# I2C SETUP
# ==============================
i2c = busio.I2C(board.SCL, board.SDA)
bus = smbus2.SMBus(1)

# ==============================
# MPU6050 SETUP
# ==============================
MPU_ADDR = 0x68

def mpu_init():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)
    print("[OK] MPU6050 ready")

def mpu_read(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)
    value = (high << 8) | low
    if value > 32768:
        value -= 65536
    return value

mpu_init()

# ==============================
# BME280 SETUP
# ==============================
try:
    bme = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)
    bme.sea_level_pressure = 1013.25
    print("[OK] BME280 ready")
    bme_ok = True
except Exception as e:
    print("[FAIL] BME280:", e)
    bme_ok = False

# ==============================
# GPS SETUP
# ==============================
try:
    gps = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1)
    print("[OK] GPS port opened")
    gps_ok = True
except:
    print("[FAIL] GPS not detected")
    gps_ok = False

# ==============================
# MAIN LOOP
# ==============================
while True:
    print("\n------------------------------")

    # ===== MPU6050 =====
    ax = mpu_read(0x3B) / 16384.0
    ay = mpu_read(0x3D) / 16384.0
    az = mpu_read(0x3F) / 16384.0

    gx = mpu_read(0x43) / 131.0
    gy = mpu_read(0x45) / 131.0
    gz = mpu_read(0x47) / 131.0

    print(f"MPU ✔ Accel[g]: X={ax:.2f} Y={ay:.2f} Z={az:.2f}")
    print(f"MPU ✔ Gyro[dps]: X={gx:.2f} Y={gy:.2f} Z={gz:.2f}")

    # ===== BME280 =====
    if bme_ok:
        print(f"BME ✔ Temp: {bme.temperature:.2f} °C")
        print(f"BME ✔ Pressure: {bme.pressure:.2f} hPa")
        print(f"BME ✔ Altitude: {bme.altitude:.2f} m")
    else:
        print("BME ✘ Not Found")

    # ===== GPS =====
    gps_fix = False
    if gps_ok and gps.in_waiting:
        try:
            line = gps.readline().decode("ascii", errors="ignore")
            if line.startswith("$GPGGA"):
                msg = pynmea2.parse(line)
                if msg.fix_quality > 0:
                    gps_fix = True
                    print(f"GPS ✔ Lat: {msg.latitude} {msg.lat_dir}")
                    print(f"GPS ✔ Lon: {msg.longitude} {msg.lon_dir}")
                    print(f"GPS ✔ Alt: {msg.altitude} m")
        except:
            pass

    if not gps_fix:
        print("GPS ✘ No Fix")

    time.sleep(1)
