import serial

ser = serial.Serial(
    "/dev/serial0",
    baudrate=115200,
    timeout=2
)

print("Listening for GPS data...")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line:
        print(line)

