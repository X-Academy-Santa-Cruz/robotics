import math
import time
import pygame
import board
import busio
from adafruit_pca9685 import PCA9685


# =========================
# USER SETTINGS
# =========================

# Thruster cant angle (degrees)
THETA_DEG = 23.3

# Joystick axis mapping
AXIS_A0_LR = 0      # strafe left/right
AXIS_A1_FB = 1      # forward/back
AXIS_A2_TWIST = 2   # yaw
AXIS_A3_POWER = 3   # master power

# Button mapping (common Xbox-like controllers; we auto-detect if these don't work)
# If your controller differs, the program will print button indices as you press them.
BUTTON_UP = 1       # "B" on many controllers (often index 1)
BUTTON_DOWN = 0     # "A" on many controllers (often index 0)

# Motor -> PCA9685 channel mapping (change if you wired differently)
MOTOR_TO_CHANNEL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# Flip direction per motor if needed (VERY common when props are mirrored)
INVERT_MOTOR = {
    1: False,
    2: False,
    3: False,
    4: False,
    5: False,
    6: False,
}

# PWM settings for ESC/servo-style control
NEUTRAL_US = 1500
MIN_US = 1100
MAX_US = 1900
ARM_SECONDS = 2.0

# Control shaping
DEADBAND = 0.08
EXPO = 1.6
UPDATE_HZ = 50

# If forward/yaw/strafe feel reversed, flip these:
INVERT_FORWARD = True
INVERT_STRAFE = False
INVERT_YAW = False

# Safety cap (0..1). Keep <1 while tuning.
MAX_CMD = 0.85


# =========================
# HELPERS
# =========================

def apply_deadband(x: float, db: float) -> float:
    if abs(x) < db:
        return 0.0
    return (x - db * (1 if x > 0 else -1)) / (1.0 - db)

def expo_curve(x: float, e: float) -> float:
    return (abs(x) ** e) * (1 if x >= 0 else -1)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def axis_to_power(a3: float) -> float:
    """
    Maps axis (-1..+1) to power (0..1).
    If your knob behaves backwards, flip it by returning (1 - p).
    """
    p = (a3 + 1.0) / 2.0
    return clamp(p, 0.0, 1.0)

def discover_buttons(js):
    """
    Print live button indices as you press them, so you can map B/A correctly.
    """
    pressed = []
    for i in range(js.get_numbuttons()):
        if js.get_button(i):
            pressed.append(i)
    return pressed


# =========================
# ROV MOTOR DRIVER + MIXER
# =========================

class ROV:
    def __init__(self):
        self.motor_to_channel = MOTOR_TO_CHANNEL
        self.invert_motor = INVERT_MOTOR

        self.neutral_us = NEUTRAL_US
        self.min_us = MIN_US
        self.max_us = MAX_US

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50

        self.stop_all()
        time.sleep(ARM_SECONDS)

        # Precompute angle factors
        theta = math.radians(THETA_DEG)
        self.c = math.cos(theta)  # forward component scale
        self.s = math.sin(theta)  # strafe component scale

    def _us_to_duty(self, us: int) -> int:
        period_us = 1_000_000 / float(self.pca.frequency)
        duty = int((us / period_us) * 65535)
        return int(clamp(duty, 0, 65535))

    def set_motor_us(self, motor: int, us: int):
        us = int(clamp(us, self.min_us, self.max_us))
        ch = self.motor_to_channel[motor]
        self.pca.channels[ch].duty_cycle = self._us_to_duty(us)

    def set_motor_throttle(self, motor: int, t: float):
        """
        t in [-1..+1], 0 = neutral
        """
        t = clamp(t, -1.0, 1.0)
        if self.invert_motor.get(motor, False):
            t *= -1.0

        span = min(self.neutral_us - self.min_us, self.max_us - self.neutral_us)
        us = int(self.neutral_us + t * span)
        self.set_motor_us(motor, us)

    def stop_all(self):
        for m in self.motor_to_channel.keys():
            self.set_motor_us(m, self.neutral_us)

    def drive(self, forward: float, strafe: float, yaw: float, vertical: float):
        """
        Commands are [-1..+1].

        Motors:
          1 = back-left (far side, odd)
          2 = back-right (close side, even)
          3 = front-left (far side, odd)
          4 = front-right (close side, even)
          5,6 = vertical
        """

        # Clamp inputs
        forward = clamp(forward, -1.0, 1.0)
        strafe  = clamp(strafe,  -1.0, 1.0)
        yaw     = clamp(yaw,     -1.0, 1.0)
        vertical= clamp(vertical,-1.0, 1.0)

        # ---- CANTED HORIZONTAL THRUST MIX ----
        # Each thruster contributes:
        #   forward component:  throttle * cos(theta)
        #   strafe component:   throttle * sin(theta)
        #
        # To solve for needed thruster outputs, we mix commands into each motor:
        #
        # Assumption (typical symmetric cant):
        # - Left side (odd motors) produce strafe LEFT when positive thrust
        # - Right side (even motors) produce strafe RIGHT when positive thrust
        #
        # If your physical strafe is reversed, flip INVERT_STRAFE or flip motor inversions.

        # Side signs: left=-1, right=+1 for strafe contribution
        side_sign = {1: -1, 2: +1, 3: -1, 4: +1}

        # Yaw: left side opposite of right side (simple differential yaw)
        yaw_sign  = {1: -1, 2: +1, 3: -1, 4: +1}

        # Compute motor throttles for 1-4
        # We divide by cos/sin scales to keep feel consistent across different angles.
        # (If theta is small, strafe authority is naturally smaller.)
        m = {}
        for motor in (1, 2, 3, 4):
            t = 0.0
            # Forward: distribute evenly; normalize by cos(theta)
            t += (forward / max(self.c, 1e-6))

            # Strafe: distribute by side; normalize by sin(theta)
            t += (strafe * side_sign[motor] / max(self.s, 1e-6))

            # Yaw: simple differential; yaw doesn’t depend on cant angle directly here
            t += (yaw * yaw_sign[motor])

            m[motor] = clamp(t, -1.0, 1.0)

        # Vertical motors 5 & 6 together
        m[5] = clamp(vertical, -1.0, 1.0)
        m[6] = clamp(vertical, -1.0, 1.0)

        # Apply
        for motor, t in m.items():
            self.set_motor_throttle(motor, t)

    def close(self):
        self.stop_all()
        time.sleep(0.2)
        self.pca.deinit()


# =========================
# JOYSTICK LOOP
# =========================

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick detected. Plug it in, then rerun.")

    js = pygame.joystick.Joystick(0)
    js.init()

    print("Joystick:", js.get_name())
    print("Axes:", js.get_numaxes(), "Buttons:", js.get_numbuttons())
    print("Mapping: A0=strafe, A1=forward, A2=yaw, A3=power (0..1)")
    print(f"Buttons: UP index={BUTTON_UP}  DOWN index={BUTTON_DOWN}")
    print("Press some buttons now; I’ll print indices I see (helps mapping).")
    print("Ctrl+C to quit.\n")

    rov = ROV()

    dt = 1.0 / UPDATE_HZ
    last_print = 0.0

    try:
        while True:
            pygame.event.pump()

            # Read axes
            a0 = js.get_axis(AXIS_A0_LR)
            a1 = js.get_axis(AXIS_A1_FB)
            a2 = js.get_axis(AXIS_A2_TWIST)
            a3 = js.get_axis(AXIS_A3_POWER)

            # Convert A3 to power scaler 0..1
            power = axis_to_power(a3)

            # Shape commands
            strafe  = expo_curve(apply_deadband(a0, DEADBAND), EXPO)
            forward = expo_curve(apply_deadband(a1, DEADBAND), EXPO)
            yaw     = expo_curve(apply_deadband(a2, DEADBAND), EXPO)

            # Optional invert
            if INVERT_STRAFE:
                strafe *= -1
            if INVERT_FORWARD:
                forward *= -1
            if INVERT_YAW:
                yaw *= -1

            # Scale by master power and cap
            strafe  = clamp(strafe  * power, -MAX_CMD, MAX_CMD)
            forward = clamp(forward * power, -MAX_CMD, MAX_CMD)
            yaw     = clamp(yaw     * power, -MAX_CMD, MAX_CMD)

            # Vertical via buttons (binary) scaled by power (smooth overall throttle still)
            up = js.get_button(BUTTON_UP) == 1
            down = js.get_button(BUTTON_DOWN) == 1
            vertical = 0.0
            if up and not down:
                vertical = +power * MAX_CMD
            elif down and not up:
                vertical = -power * MAX_CMD

            # Drive
            rov.drive(forward=forward, strafe=strafe, yaw=yaw, vertical=vertical)

            # Debug prints + button discovery
            now = time.time()
            if now - last_print > 0.25:
                last_print = now
                pressed = discover_buttons(js)
                if pressed:
                    print(f"\nButtons pressed (indices): {pressed}")

                print(
                    f"\rA0={a0:+.2f} A1={a1:+.2f} A2={a2:+.2f} A3={a3:+.2f}  "
                    f"power={power:.2f}  "
                    f"FWD={forward:+.2f} STR={strafe:+.2f} YAW={yaw:+.2f} VERT={vertical:+.2f}      ",
                    end=""
                )

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        rov.close()
        pygame.quit()
        print("\nExited cleanly.")


if __name__ == "__main__":
    main()
