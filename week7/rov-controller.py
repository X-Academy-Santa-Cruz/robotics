import pickle
import socket

import pi_servo_hat

servo_hat = pi_servo_hat.PiServoHat()
servo_hat.restart()


LOCAL_IP = "0.0.0.0"
LOCAL_PORT = 20001
BUFFER_SIZE = 128
MESSAGE_FROM_SERVER = "Hello UDP Client"
BYTES_TO_SEND = str.encode(MESSAGE_FROM_SERVER)

# Create a datagram socket
udp_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
udp_server_socket.bind((LOCAL_IP, LOCAL_PORT))

print("UDP server up and listening")

# Listen for incoming datagrams
count = 0
while True:

    bytes_address_pair = udp_server_socket.recvfrom(BUFFER_SIZE)

    message = bytes_address_pair[0]

    address = bytes_address_pair[1]

    client_message = "Message from Client:{}".format(message)
    client_ip = "Client IP Address:{}".format(address)

    if client_ip == "192.168.1.47":
        print("from main computer")

    data = pickle.loads(message)
    wheel_scaled_left = data[0]
    wheel_scaled_right = data[1]
    servo_hat.move_servo_position(0, wheel_scaled_left)
    servo_hat.move_servo_position(1, wheel_scaled_right)

    print(count, data)
    count += 1

    udp_server_socket.sendto(str.encode("test"), address)
