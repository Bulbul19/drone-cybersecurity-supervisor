import time
import math
import serial
import numpy as np
import smbus2
# ==============================
# OPTIONAL: SCIKIT-FUZZY
# ==============================
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except:
    SKFUZZY_AVAILABLE = False

# ==============================
# MPU6050 (SMBUS)
# ==============================
MPU_ADDR = 0x68
bus = smbus2.SMBus(1)

def init_mpu():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)

def read_word(reg):
    h = bus.read_byte_data(MPU_ADDR, reg)
    l = bus.read_byte_data(MPU_ADDR, reg+1)
    v = (h << 8) + l
    return v - 65536 if v > 32768 else v

def read_mpu():
    ax = read_word(0x3B)/16384.0
    ay = read_word(0x3D)/16384.0
    az = read_word(0x3F)/16384.0
    gx = read_word(0x43)/131.0
    gy = read_word(0x45)/131.0
    gz = read_word(0x47)/131.0
    def read_mpu():
    ax = read_word(0x3B)/16384.0
    ay = read_word(0x3D)/16384.0
    az = read_word(0x3F)/16384.0

    gx = read_word(0x43)/131.0
    gy = read_word(0x45)/131.0
    gz = read_word(0x47)/131.0

    accel_mag = math.sqrt(ax*ax + ay*ay + az*az)
    max_accel = accel_mag

    mean_gyro = (abs(gx) + abs(gy) + abs(gz)) / 3.0

    return max_accel, mean_gyro
# ==============================
# BME280 (ALTITUDE SOURCE) – FIXED
# ==============================
import board
import busio
from adafruit_bme280 import basic as adafruit_bme280
from collections import deque

i2c = busio.I2C(board.SCL, board.SDA)
bme = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)
bme.sea_level_pressure = 1008.0  # safer default

bme_alt_window = deque(maxlen=5)

def get_bme_altitude():
    try:
        alt = float(bme.altitude)
        bme_alt_window.append(alt)
        return sum(bme_alt_window)/len(bme_alt_window)
    except Exception:
        return None

# ==============================
# GPS NEO-6M
# ==============================
gps = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1)

def read_gps():
    for _ in range(10):
        line = gps.readline().decode(errors="ignore")
        if "$GPGGA" in line:
            p = line.split(",")
            if p[2] and p[4]:
                sats = int(p[7])
                hdop = float(p[8])
                alt = float(p[9])
                return sats, hdop, alt
    return None, None, None

# ==============================
# FUZZY ENGINE
# ==============================
def build_fuzzy_engine():
    satellites = ctrl.Antecedent(np.arange(0,21,1),'satellites')
    hdop = ctrl.Antecedent(np.arange(0,21,0.1),'hdop')
    jump = ctrl.Antecedent(np.arange(0,101,1),'jump')
    max_accel = ctrl.Antecedent(np.arange(0,5.1,0.1),'max_accel')
    mean_gyro = ctrl.Antecedent(np.arange(0,301,1),'mean_gyro')
    vib = ctrl.Antecedent(np.arange(0,2.01,0.01),'vib')
    alt_err = ctrl.Antecedent(np.arange(0,51,0.5),'alt_err')
    trust = ctrl.Consequent(np.arange(0,1.01,0.01),'trust')

    satellites['FEW']=fuzz.trimf(satellites.universe,[0,0,6])
    satellites['MANY']=fuzz.trimf(satellites.universe,[10,20,20])

    hdop['GOOD']=fuzz.trimf(hdop.universe,[0,1,2])
    hdop['POOR']=fuzz.trimf(hdop.universe,[2,20,20])

    jump['OK']=fuzz.trimf(jump.universe,[0,0,5])
    jump['BAD']=fuzz.trimf(jump.universe,[10,100,100])

    max_accel['LOW']=fuzz.trimf(max_accel.universe,[0,0,1])
    max_accel['HIGH']=fuzz.trimf(max_accel.universe,[2,5,5])

    mean_gyro['STABLE']=fuzz.trimf(mean_gyro.universe,[0,0,30])
    mean_gyro['FAST']=fuzz.trimf(mean_gyro.universe,[80,300,300])

    vib['LOW']=fuzz.trimf(vib.universe,[0,0,0.1])
    vib['HIGH']=fuzz.trimf(vib.universe,[0.4,2,2])

    alt_err['SMALL']=fuzz.trimf(alt_err.universe,[0,0,3])
    alt_err['LARGE']=fuzz.trimf(alt_err.universe,[5,50,50])

    trust['LOW']=fuzz.trimf(trust.universe,[0,0,0.4])
    trust['HIGH']=fuzz.trimf(trust.universe,[0.6,1,1])

    rules=[
        ctrl.Rule(jump['BAD'] | alt_err['LARGE'], trust['LOW']),
        ctrl.Rule(satellites['MANY'] & hdop['GOOD'] &
                  max_accel['LOW'] & mean_gyro['STABLE'] &
                  vib['LOW'] & alt_err['SMALL'], trust['HIGH'])
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)

# ==============================
# MAIN
# ==============================
init_mpu()
fuzzy = build_fuzzy_engine()

print("\n=== SENSOR FUSION STARTED ===")

prev_alt = None

while True:
    ax, ay, az, accel_mag, gyro_mag = read_mpu()
    sats, hdop, gps_alt = read_gps()
    bme_alt = bme.altitude

    if gps_alt is None:
        jump = 0
        alt_err = 0
    else:
        jump = abs(gps_alt - prev_alt) if prev_alt else 0
        alt_err = abs(bme_alt - gps_alt)
        prev_alt = gps_alt

    vib = abs(accel_mag - 1.0)

    fuzzy.reset()
    fuzzy.input['satellites'] = sats or 0
    fuzzy.input['hdop'] = hdop or 20
    fuzzy.input['jump'] = jump
    fuzzy.input['max_accel'] = accel_mag
    fuzzy.input['mean_gyro'] = gyro_mag
    fuzzy.input['vib'] = vib
    fuzzy.input['alt_err'] = alt_err
    fuzzy.compute()


    trust = fuzzy.output.get('trust', 0.0)
    print(f"TRUST SCORE: {trust:.2f}")

    print("\n------------------------------")
    print(f"MPU Accel Mag: {accel_mag:.2f} g")
    print(f"Gyro Mag: {gyro_mag:.1f} dps")
    print(f"BME Alt: {bme_alt:.2f} m")
    print(f"GPS Alt: {gps_alt}")
    print(f"Alt Error: {alt_err:.2f} m")
    print(f"TRUST SCORE: {trust:.2f}")

    if trust < 0.4:
        print("🚨 GPS SPOOFING / SENSOR FAULT SUSPECTED")

    time.sleep(2)
