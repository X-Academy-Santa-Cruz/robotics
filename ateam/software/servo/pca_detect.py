import board, busio
from adafruit_pca9685 import PCA9685

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)

print("PCA9685 address:", hex(pca.i2c_device.device_address))
pca.frequency = 50
print("Frequency set to", pca.frequency)

pca.deinit()
print("DONE – hardware detected")
