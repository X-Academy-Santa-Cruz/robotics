import time
import board
import busio
from adafruit_pca9685 import PCA9685

# =========================
# SETTINGS YOU MAY TWEAK
# =========================

# Motor number -> PCA9685 channel
MOTOR_TO_CHANNEL = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6
}

FREQ_HZ = 50
NEUTRAL_US = 1500
MIN_US = 1100
MAX_US = 1900

SLOW_FWD_US = 1600
FAST_FWD_US = 1800
SLOW_REV_US = 1400
FAST_REV_US = 1200

ARM_SECONDS = 2.0
STEP_SECONDS = 1.0
PAUSE_BETWEEN_MOTORS = 0.8

INVERT = {1: False, 2: False, 3: False, 4: False, 5: False, 6: False}


# =========================
# PCA9685 UTILITIES
# =========================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def us_to_duty(us: int, freq_hz: float) -> int:
    period_us = 1_000_000 / freq_hz
    duty = int((us / period_us) * 65535)
    return clamp(duty, 0, 65535)


def set_channel_us(pca: PCA9685, channel: int, us: int):
    pca.channels[channel].duty_cycle = us_to_duty(us, pca.frequency)


def set_motor_us(pca: PCA9685, motor: int, us: int):
    if motor not in MOTOR_TO_CHANNEL:
        return
    ch = MOTOR_TO_CHANNEL[motor]
    us = clamp(int(us), MIN_US, MAX_US)
    set_channel_us(pca, ch, us)


def stop_all(pca: PCA9685):
    for m in MOTOR_TO_CHANNEL:
        set_motor_us(pca, m, NEUTRAL_US)


def maybe_invert_us(motor: int, us: int) -> int:
    if not INVERT.get(motor, False):
        return us
    return int(NEUTRAL_US - (us - NEUTRAL_US))


# =========================
# TEST SEQUENCES
# =========================

def run_step(pca: PCA9685, motor: int, label: str, us: int, seconds: float):
    channel = MOTOR_TO_CHANNEL[motor]
    us = maybe_invert_us(motor, us)
    print(f"Motor {motor} (Channel {channel}): {label} @ {us}us")
    set_motor_us(pca, motor, us)
    time.sleep(seconds)


def test_single_motor(pca, motor):
    """Runs the full test sequence for just one motor."""
    if motor not in MOTOR_TO_CHANNEL:
        print(f"Error: Motor {motor} is not defined in MOTOR_TO_CHANNEL.")
        return

    print("\n==============================")
    print(f"Testing motor {motor} (channel {MOTOR_TO_CHANNEL[motor]})")
    print("==============================")

    stop_all(pca)
    time.sleep(0.2)

    run_step(pca, motor, "FAST FORWARD", FAST_FWD_US, STEP_SECONDS)
    run_step(pca, motor, "SLOW FORWARD", SLOW_FWD_US, STEP_SECONDS)
    run_step(pca, motor, "SLOW REVERSE", SLOW_REV_US, STEP_SECONDS)
    run_step(pca, motor, "FAST REVERSE", FAST_REV_US, STEP_SECONDS)
    run_step(pca, motor, "STOP", NEUTRAL_US, 0.5)


def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = FREQ_HZ

    try:
        print("Arming: sending NEUTRAL to all motors...")
        stop_all(pca)
        time.sleep(ARM_SECONDS)

        print("\n--- TEST MENU ---")
        print("0: Test ALL motors")
        print("1-6: Test a SPECIFIC motor")
        choice = input("Enter your choice: ")

        if choice == "0":
            # Test all defined motors
            for motor in sorted(MOTOR_TO_CHANNEL.keys()):
                test_single_motor(pca, motor)
                time.sleep(PAUSE_BETWEEN_MOTORS)
        else:
            # Test specific motor
            m_num = int(choice)
            test_single_motor(pca, m_num)

        print("\nTesting complete. Stopping all.")
        stop_all(pca)

    except KeyboardInterrupt:
        print("\nInterrupted. Stopping all.")
        stop_all(pca)
    except ValueError:
        print("Invalid input. Please enter a number.")
    finally:
        time.sleep(0.2)
        pca.deinit()
        print("Done.")


if __name__ == "__main__":
    main()