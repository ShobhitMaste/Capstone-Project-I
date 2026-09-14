"""Tests for mock servo controller."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestMockServo:
    """Verify MockServoController state transitions."""

    def _make_servo(self):
        from fedtrap.servo_controller import MockServoController
        return MockServoController(
            home_angle=90,
            capture_angle=150,
            release_angle=30,
        )

    def test_initial_state_home(self):
        servo = self._make_servo()
        assert servo.get_state() == "HOME"

    def test_capture_moves_to_capture(self):
        servo = self._make_servo()
        servo.capture()
        assert servo.get_state() == "CAPTURE"

    def test_release_moves_to_release(self):
        servo = self._make_servo()
        servo.release()
        assert servo.get_state() == "RELEASE"

    def test_home_returns_to_home(self):
        servo = self._make_servo()
        servo.capture()
        servo.home()
        assert servo.get_state() == "HOME"

    def test_emergency_stop_returns_home(self):
        servo = self._make_servo()
        servo.capture()
        servo.emergency_stop()
        assert servo.get_state() == "HOME"

    def test_sequence_capture_release_home(self):
        servo = self._make_servo()
        servo.capture()
        assert servo.get_state() == "CAPTURE"
        servo.release()
        assert servo.get_state() == "RELEASE"
        servo.home()
        assert servo.get_state() == "HOME"

    def test_cleanup(self):
        servo = self._make_servo()
        servo.capture()
        servo.cleanup()
        assert servo.get_state() == "HOME"

    def test_get_angle_values(self):
        servo = self._make_servo()
        assert servo.get_current_angle() == 90  # home
        servo.capture()
        assert servo.get_current_angle() == 150
        servo.release()
        assert servo.get_current_angle() == 30
