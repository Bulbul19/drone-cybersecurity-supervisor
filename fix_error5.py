init_bme280()
while True:
    print("Baro Alt:", get_baro_altitude())
    time.sleep(1)
