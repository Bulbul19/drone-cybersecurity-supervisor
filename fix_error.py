import time
import serial
import math
import smbus2

# ==============================
# GPS NEO-6M
# ==============================
def read_gps():
    try:
        ser = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1)
        for _ in range(10):
            line = ser.readline().decode(errors="ignore")
            if "$GPGGA" in line:
                parts = line.split(",")
                if parts[2] and parts[4]:
                    lat = float(parts[2])
                    lon = float(parts[4])
                    sats = int(parts[7])
                    hdop = float(parts[8])
                    alt = float(parts[9])
                    ser.close()
                    return lat, lon, sats, hdop, alt
        ser.close()
    except Exception as e:
        print("GPS Error:", e)
    return None

# ==============================
# MPU6050
# ==============================
MPU_ADDR = 0x68
bus = smbus2.SMBus(1)

def init_mpu():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)

def read_word(addr):
    high = bus.read_byte_data(MPU_ADDR, addr)
    low = bus.read_byte_data(MPU_ADDR, addr+1)
    val = (high << 8) + low
    return val - 65536 if val > 32768 else val

def read_mpu():
    try:
        ax = read_word(0x3B)/16384.0
        ay = read_word(0x3D)/16384.0
        az = read_word(0x3F)/16384.0
        gx = read_word(0x43)/131.0
        gy = read_word(0x45)/131.0
        gz = read_word(0x47)/131.0
        return ax, ay, az, gx, gy, gz
    except Exception as e:
        print("MPU Error:", e)
        return None

# ==============================
# BME280  (not BMP280)
# ==============================
try:
    import board
    import busio
    import adafruit_bme280

    i2c = busio.I2C(board.SCL, board.SDA)   # ✅ THIS LINE WAS MISSING

    bmp = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)
    bmp.sea_level_pressure = 1013.25
    bmp_ok = True

except Exception as e:
    print("BME280 Error:", e)
    bmp_ok = False

# ==============================
# MAIN LOOP
# ==============================
def main():
    print("\n===== SENSOR HARDWARE TEST =====\n")

    try:
        init_mpu()
        print("MPU6050 initialized ✔")
    except:
        print("MPU6050 init failed ✘")

    while True:
        print("\n------------------------------")

        # GPS
        gps = read_gps()
        if gps:
            lat, lon, sats, hdop, alt = gps
            print(f"GPS ✔ Lat:{lat} Lon:{lon}")
            print(f"Satellites:{sats} HDOP:{hdop} Alt:{alt}m")
        else:
            print("GPS ✘ No Fix")

        # MPU
        mpu = read_mpu()
        if mpu:
            ax, ay, az, gx, gy, gz = mpu
            print(f"MPU ✔ Accel[g]: X={ax:.2f} Y={ay:.2f} Z={az:.2f}")
            print(f"MPU ✔ Gyro[dps]: X={gx:.1f} Y={gy:.1f} Z={gz:.1f}")
        else:
            print("MPU ✘ No Data")

        # BMP280
        if bmp_ok:
            try:
                print(f"BMP280 ✔ Temp:{bmp.temperature:.2f}°C")
                print(f"BMP280 ✔ Pressure:{bmp.pressure:.2f} hPa")
                print(f"BMP280 ✔ Altitude:{bmp.altitude:.2f} m")
            except:
                print("BMP280 ✘ Read Error")
        else:
            print("BMP280 ✘ Not Found")

        time.sleep(2)

if __name__ == "__main__":
    main()
