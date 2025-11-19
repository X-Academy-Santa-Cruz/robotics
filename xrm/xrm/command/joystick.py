import click
from math import atan2, degrees, pi, sin, sqrt
from time import sleep, time
import pygame

# noinspection PyUnusedLocal
@click.command("joystick")
def command_joystick():
    """Read joystick values.
    """

    pygame.display.init()
    pygame.joystick.init()
    controller = pygame.joystick.Joystick(0)
    controller.init()

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

        jx = -controller.get_axis(1)
        jy = controller.get_axis(0)
        all_stop = controller.get_button(1)
        theta = atan2(jy, jx)
        if abs(theta) > pi / 2.0:
            direction = -1.0
        else:
            direction = 1.0

        translation = direction * sqrt(jx * jx + jy * jy)
        velocity_translation = translation
        velocity_rotation = direction * sin(theta)
        wheel_velocity_right = (mt * velocity_translation - mr * velocity_rotation) / (mt + mr)
        wheel_velocity_left = (mt * velocity_translation + mr * velocity_rotation) / (mt + mr)

        wheel_scaled_right = wheel_velocity_right * 54.0 + 45.0
        wheel_scaled_left =  wheel_velocity_left * 54.0 + 45.0

        print(
            'time: {:+9.3f} jx: {:+6.3f} jy: {:+6.3f} theta: {:+6.1f} vt: {:+6.3f} vr: {:+6.3f} wr: {:+6.3f} '
            'wl: {:+6.3f} sr: {:+6.1f} sl: {:+6.2f} '.format(
                time(), jx, jy, degrees(theta), velocity_translation, velocity_rotation, wheel_velocity_right,
                wheel_velocity_left, wheel_scaled_right, wheel_scaled_left))

        time_1 = time()
        delta = time_1 - time_0
        delay = max(0.0, interval - (delta + 0.00079))

        sleep(delay)
        time_0 = time()
