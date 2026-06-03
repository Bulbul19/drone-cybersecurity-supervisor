import serial

try:
    ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
    print("Port opened")
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(line)
except Exception as e:
    print("Error:", e)
