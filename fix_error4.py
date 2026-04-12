import time
import math
import serial
import numpy as np
import smbus2
import skfuzzy as fuzz
from skfuzzy import control as ctrl
last_good_pressure = None
last_good_altitude = None

# ==============================
# CONFIG
# ==============================
MPU_ADDR = 0x68
BME_ADDR = 0x76
I2C_BUS = 1

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 9600

# ==============================
# I2C
# ==============================
bus = smbus2.SMBus(I2C_BUS)

# ==============================
# MPU6050
# ==============================
def init_mpu():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)

def read_word(reg):
    h = bus.read_byte_data(MPU_ADDR, reg)
    l = bus.read_byte_data(MPU_ADDR, reg + 1)
    v = (h << 8) + l
    return v - 65536 if v > 32768 else v

def read_mpu():
    ax = read_word(0x3B) / 16384.0
    ay = read_word(0x3D) / 16384.0
    az = read_word(0x3F) / 16384.0

    gx = read_word(0x43) / 131.0
    gy = read_word(0x45) / 131.0
    gz = read_word(0x47) / 131.0

    accel_mag = math.sqrt(ax*ax + ay*ay + az*az)
    accel_vib = abs(accel_mag - 1.0)
    gyro_mag = abs(gx) + abs(gy) + abs(gz)

    return accel_vib, gyro_mag

# ==============================
# BME280
# ==============================
def init_bme280():
    bus.write_byte_data(BME_ADDR, 0xF2, 0x01)
    bus.write_byte_data(BME_ADDR, 0xF4, 0x27)
    bus.write_byte_data(BME_ADDR, 0xF5, 0xA0)

def read_pressure_raw():
    try:
        data = bus.read_i2c_block_data(BME_ADDR, 0xF7, 6)
        adc_p = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        return adc_p
    except OSError as e:
        print("⚠️ BME280 I2C read error, using last value")
        return None

def pressure_to_altitude(p_hpa, p0=1013.25):
    return 44330.0 * (1.0 - (p_hpa / p0) ** 0.1903)

# ==============================
# GPS
# ==============================
gps = serial.Serial('/dev/ttyAMA0', 9600, timeout=0.1)
def read_gps_altitude(gps):
    if gps.in_waiting == 0:
        return None

    try:
        line = gps.readline().decode(errors="ignore").strip()
        if line.startswith("$GPGGA"):
            parts = line.split(",")
            if len(parts) > 9 and parts[9]:
                return float(parts[9])
    except Exception:
        pass

    return None
# ==============================
# FUZZY SYSTEM (FIXED)
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

    satellites['FEW'] = fuzz.trimf(satellites.universe,[0,0,6])
    satellites['MANY'] = fuzz.trimf(satellites.universe,[10,20,20])

    hdop['GOOD'] = fuzz.trimf(hdop.universe,[0,1,2])
    hdop['POOR'] = fuzz.trimf(hdop.universe,[2,20,20])

    jump['OK'] = fuzz.trimf(jump.universe,[0,0,5])
    jump['BAD'] = fuzz.trimf(jump.universe,[10,100,100])

    max_accel['LOW'] = fuzz.trimf(max_accel.universe,[0,0,1])
    max_accel['HIGH'] = fuzz.trimf(max_accel.universe,[2,5,5])

    mean_gyro['STABLE'] = fuzz.trimf(mean_gyro.universe,[0,0,30])
    mean_gyro['FAST'] = fuzz.trimf(mean_gyro.universe,[80,300,300])

    vib['LOW'] = fuzz.trimf(vib.universe,[0,0,0.1])
    vib['HIGH'] = fuzz.trimf(vib.universe,[0.4,2,2])

    alt_err['SMALL'] = fuzz.trimf(alt_err.universe,[0,0,3])
    alt_err['LARGE'] = fuzz.trimf(alt_err.universe,[5,50,50])

    trust['LOW'] = fuzz.trimf(trust.universe,[0,0,0.4])
    trust['HIGH'] = fuzz.trimf(trust.universe,[0.6,1,1])

    rules = [
            ctrl.Rule(jump['BAD'] | alt_err['LARGE'], trust['LOW']),
            ctrl.Rule(satellites['MANY'] & hdop['GOOD'] &
                      max_accel['LOW'] & mean_gyro['STABLE'] &
                      vib['LOW'] & alt_err['SMALL'], trust['HIGH']),
            ctrl.Rule(satellites['FEW'], trust['LOW'])  # fallback rule
         ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim   # ✅ THIS WAS THE PROBLEM

# ==============================
# MAIN LOOP
# ==============================
print("Loop alive...")
init_mpu()
init_bme280()

prev_gps_alt = None
prev_baro_alt = None
last_good_pressure = None
last_good_altitude = None
print("\n=== HYBRID TRUST MONITOR STARTED ===\n")
fuzzy = build_fuzzy_engine()
print("FUZZY ENGINE READY:", fuzzy, flush=True)

while True:
    print("STEP 1: loop entered", flush=True)

    print("STEP 2: before IMU", flush=True)
    accel_vib, gyro_mag = read_mpu()
    print("STEP 3: after IMU", flush=True)

    print("STEP 4: before BARO", flush=True)
    raw_p = read_pressure_raw()
    p_hpa = raw_p / 25600.0
    baro_alt = pressure_to_altitude(p_hpa)
    print("STEP 5: after BARO", flush=True)

    print("STEP 6: before GPS", flush=True)
    gps_alt = read_gps_altitude(gps)
    print("STEP 7: after GPS", gps_alt, flush=True)

    # ==============================
    # ALTITUDE ERROR (GPS vs BARO)
    # ==============================
    if gps_alt is None or prev_gps_alt is None:
        delta_gps = 0.0
    else:
        delta_gps = abs(gps_alt - prev_gps_alt)

    if prev_baro_alt is None:
        delta_baro = 0.0
    else:
        delta_baro = abs(baro_alt - prev_baro_alt)

    alt_err = abs(delta_gps - delta_baro)
    alt_err = min(50.0, alt_err)

    prev_gps_alt = gps_alt
    prev_baro_alt = baro_alt

  # ---------- SAFE PLACEHOLDERS ----------
    satellites = 0
    hdop = 20.0
    jump = 0.0
    # --------------------------------------

    print("STEP 8: before FUZZY", flush=True)

    fuzzy.reset()

    fuzzy.input['satellites'] = satellites
    fuzzy.input['hdop'] = hdop
    fuzzy.input['jump'] = delta_gps
    fuzzy.input['max_accel'] = accel_vib
    fuzzy.input['mean_gyro'] = gyro_mag
    fuzzy.input['vib'] = accel_vib
    fuzzy.input['alt_err'] = alt_err

    fuzzy.compute()

    if 'trust' in fuzzy.output:
        trust = fuzzy.output['trust']
    else:
        trust = 0.0

    print(f"TRUST SCORE: {trust:.2f}")
    print("-" * 60)

    time.sleep(1)
