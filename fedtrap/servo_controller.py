"""FedTrap servo controller — hardware abstraction.

Provides MockServoController for desktop testing and PiServoController
for real Raspberry Pi GPIO control.  The factory ``create_servo()``
auto-detects the platform.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Abstract base
# =========================================================================

class ServoController:
    """Interface for servo gate control."""

    def home(self) -> None:
        raise NotImplementedError

    def capture(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def emergency_stop(self) -> None:
        raise NotImplementedError

    def get_state(self) -> str:
        raise NotImplementedError

    def get_current_angle(self) -> float:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError


# =========================================================================
# Mock (desktop / testing)
# =========================================================================

class MockServoController(ServoController):
    """Simulated servo — logs actions, tracks state, no hardware."""

    def __init__(
        self,
        home_angle: float = 90,
        capture_angle: float = 150,
        release_angle: float = 30,
    ):
        self._home_angle = home_angle
        self._capture_angle = capture_angle
        self._release_angle = release_angle
        self._state = "HOME"
        self._angle = home_angle
        logger.info("MockServoController initialised (home=%s°)", home_angle)

    def home(self) -> None:
        self._state = "HOME"
        self._angle = self._home_angle
        logger.info("MockServo → HOME (%s°)", self._angle)

    def capture(self) -> None:
        self._state = "CAPTURE"
        self._angle = self._capture_angle
        logger.info("MockServo → CAPTURE (%s°)", self._angle)

    def release(self) -> None:
        self._state = "RELEASE"
        self._angle = self._release_angle
        logger.info("MockServo → RELEASE (%s°)", self._angle)

    def emergency_stop(self) -> None:
        self._state = "HOME"
        self._angle = self._home_angle
        logger.warning("MockServo → EMERGENCY STOP → HOME (%s°)", self._angle)

    def get_state(self) -> str:
        return self._state

    def get_current_angle(self) -> float:
        return self._angle

    def cleanup(self) -> None:
        self.home()
        logger.info("MockServo cleaned up")


# =========================================================================
# Real Pi servo
# =========================================================================

class PiServoController(ServoController):
    """SG90 servo on Raspberry Pi GPIO via pigpio or RPi.GPIO fallback."""

    def __init__(
        self,
        gpio: int = 18,
        home_angle: float = 90,
        capture_angle: float = 150,
        release_angle: float = 30,
        use_pigpio: bool = True,
    ):
        self._gpio = gpio
        self._home_angle = home_angle
        self._capture_angle = capture_angle
        self._release_angle = release_angle
        self._use_pigpio = use_pigpio
        self._state = "HOME"
        self._angle = home_angle

        if self._use_pigpio:
            try:
                import pigpio  # type: ignore
                self._pi = pigpio.pi()
                if not self._pi.connected:
                    raise RuntimeError("pigpio daemon not running")
                logger.info("PiServo using pigpio on GPIO %d", gpio)
            except (ImportError, RuntimeError):
                logger.warning("pigpio unavailable — falling back to RPi.GPIO")
                self._use_pigpio = False

        if not self._use_pigpio:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._gpio, GPIO.OUT)
            self._pwm = GPIO.PWM(self._gpio, 50)
            self._pwm.start(0)
            logger.info("PiServo using RPi.GPIO PWM on GPIO %d", gpio)

        self._set_angle(home_angle)

    # --- angle control ---
    def _set_angle(self, angle: float) -> None:
        self._angle = angle
        if self._use_pigpio:
            pw = 500 + (angle / 180.0) * 2000
            self._pi.set_servo_pulsewidth(self._gpio, int(pw))
        else:
            duty = angle / 18.0 + 2.0
            self._pwm.ChangeDutyCycle(duty)
            time.sleep(0.3)
            self._pwm.ChangeDutyCycle(0)

    def home(self) -> None:
        self._set_angle(self._home_angle)
        self._state = "HOME"

    def capture(self) -> None:
        self._set_angle(self._capture_angle)
        self._state = "CAPTURE"

    def release(self) -> None:
        self._set_angle(self._release_angle)
        self._state = "RELEASE"

    def emergency_stop(self) -> None:
        if self._use_pigpio:
            self._pi.set_servo_pulsewidth(self._gpio, 0)
        else:
            self._pwm.ChangeDutyCycle(0)
        self._state = "HOME"
        self._angle = self._home_angle
        logger.warning("PiServo → EMERGENCY STOP")

    def get_state(self) -> str:
        return self._state

    def get_current_angle(self) -> float:
        return self._angle

    def cleanup(self) -> None:
        self.home()
        if self._use_pigpio:
            self._pi.set_servo_pulsewidth(self._gpio, 0)
            self._pi.stop()
        else:
            import RPi.GPIO as GPIO  # type: ignore
            self._pwm.stop()
            GPIO.cleanup()
        logger.info("PiServo cleaned up")


# =========================================================================
# Factory
# =========================================================================

def create_servo(config: Optional[Dict[str, Any]] = None) -> ServoController:
    """Create the appropriate servo controller for the current platform."""
    if config is None:
        from fedtrap.config import get_config
        cfg = get_config()
        config = cfg.get("servo", {})

    gpio = config.get("gpio", 18)
    home = config.get("home_angle", 90)
    cap = config.get("capture_angle", 150)
    rel = config.get("release_angle", 30)
    use_pigpio = config.get("use_pigpio", True)

    try:
        import RPi.GPIO  # type: ignore  # noqa: F401
        return PiServoController(gpio, home, cap, rel, use_pigpio)
    except ImportError:
        logger.info("RPi.GPIO not available — using MockServoController")
        return MockServoController(home, cap, rel)
