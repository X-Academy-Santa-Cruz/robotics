from math import atan2, degrees, pi, sin, sqrt
from time import sleep, time

# import pi_servo_hat
import pygame
import socket
import pickle


def clamp(x, in_min, in_max):
    return max(in_min, min(x, in_max))


def zero_box(n):
    left = -0.1
    right = 0.1
    if (n > left) and (n < right):
        return 0.0
    return n


ip = "10.42.0.1"  # ROV IP!!!
server_address_port = (ip, 20001)

bufferSize = 128

udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

send_message = pickle.dumps([1500, 1500, 1500, 1500,
                             1500, 1500, 1500, 1500,
                             1500, 1500, 1500, 1500,
                             1500, 1500, 1500, 1500])

pygame.display.init()
pygame.joystick.init()
controller = pygame.joystick.Joystick(0)
controller.init()

# servo_hat = pi_servo_hat.PiServoHat()
# servo_hat.restart()

time_0 = time()
time_1 = time()

i = 0
frequency = 20.0
interval = 1.0 / frequency
delay = 0.1
delta = 0.0

mt = 2.0  # translation velocity multiplier
mr = 1.0  # rotation velocity multiplier
theta = 0.0
velocity_translation = 0.0
velocity_rotation = 0.0

while True:
    pygame.event.pump()

    i += 1

    jx = zero_box(-controller.get_axis(1))
    jy = zero_box(controller.get_axis(0))
    all_stop = controller.get_button(1)
    if all_stop == 1:
        wheel_velocity_right = 0.0
        wheel_velocity_left = 0.0
    else:
        theta = atan2(jy, jx)
        if abs(theta) > pi / 2.0:
            direction = -1.0
        else:
            direction = 1.0

        translation = direction * sqrt(jx * jx + jy * jy)
        velocity_translation = clamp(translation, -1.0, 1.0)
        velocity_rotation = direction * sin(theta)
        wheel_velocity_right = (mt * velocity_translation - mr * velocity_rotation) / (mt + mr)
        wheel_velocity_left = (mt * velocity_translation + mr * velocity_rotation) / (mt + mr)

    wheel_scaled_right = clamp(wheel_velocity_right * 54.0 + 45.0, 0.0, 90.0)
    wheel_scaled_left = clamp(wheel_velocity_left * 54.0 + 45.0, 0.0, 90.0)

    print(
        'time: {:+9.3f} jx: {:+6.3f} jy: {:+6.3f} theta: {:+6.1f} vt: {:+6.3f} vr: {:+6.3f} wr: {:+6.3f} '
        'wl: {:+6.3f} sr: {:+6.1f} sl: {:+6.2f} '.format(
            time(), jx, jy, degrees(theta), velocity_translation, velocity_rotation, wheel_velocity_right,
            wheel_velocity_left, wheel_scaled_right, wheel_scaled_left))

    # servo_hat.move_servo_position(0, wheel_scaled_left)
    # servo_hat.move_servo_position(1, wheel_scaled_right)

    send_message = pickle.dumps([wheel_scaled_left, wheel_scaled_right, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0])

    try:
        udp_client_socket.sendto(send_message, server_address_port)
    except :
        print("Connection Lost, reconnecting...")
        UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    time_1 = time()
    delta = time_1 - time_0
    delay = max(0.0, interval - (delta + 0.00079))

    sleep(delay)
    time_0 = time()
