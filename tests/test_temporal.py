"""Tests for temporal confirmation state machine."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestTemporalFilter:
    """Verify multi-frame confirmation FSM."""

    def _make_filter(self, frames=3, hold=0.5):
        from fedtrap.temporal_filter import TemporalFilter
        return TemporalFilter(
            confirmation_frames=frames,
            hold_duration=hold,
            confidence_threshold=0.85,
        )

    # --- Basic state transitions ---

    def test_initial_state_idle(self):
        tf = self._make_filter()
        assert tf.get_state() == "IDLE"

    def test_single_detection_becomes_candidate(self):
        tf = self._make_filter()
        result = tf.update("grasshopper", 0.91)
        assert result["state"] in ("CANDIDATE", "IDLE")

    def test_three_consecutive_same_class_confirmed(self):
        tf = self._make_filter(frames=3)
        tf.update("grasshopper", 0.91)
        tf.update("grasshopper", 0.94)
        result = tf.update("grasshopper", 0.93)
        assert result["state"] == "CONFIRMED"
        assert result["class_name"] == "grasshopper"

    def test_mixed_predictions_reset(self):
        """Different class predictions should prevent confirmation."""
        tf = self._make_filter(frames=3)
        tf.update("grasshopper", 0.91)
        tf.update("butterfly", 0.88)
        result = tf.update("grasshopper", 0.90)
        assert result["state"] != "CONFIRMED"

    def test_low_confidence_resets(self):
        """Below-threshold predictions should not contribute to confirmation."""
        tf = self._make_filter(frames=3)
        tf.update("grasshopper", 0.91)
        tf.update("grasshopper", 0.60)  # below threshold
        result = tf.update("grasshopper", 0.93)
        assert result["state"] != "CONFIRMED"

    # --- Reset ---

    def test_reset_returns_to_idle(self):
        tf = self._make_filter()
        tf.update("grasshopper", 0.91)
        tf.reset()
        assert tf.get_state() == "IDLE"

    # --- Edge cases ---

    def test_confirmation_with_one_frame(self):
        """If confirmation_frames=1, a single strong prediction confirms."""
        tf = self._make_filter(frames=1)
        result = tf.update("beetle", 0.95)
        assert result["state"] == "CONFIRMED"
        assert result["class_name"] == "beetle"

    def test_beneficial_insect_confirmed(self):
        tf = self._make_filter(frames=3)
        tf.update("honeybee", 0.92)
        tf.update("honeybee", 0.90)
        result = tf.update("honeybee", 0.91)
        assert result["state"] == "CONFIRMED"
        assert result["class_name"] == "honeybee"

    def test_hold_state_after_confirmation(self):
        """After confirmation + actuation, filter should enter HOLD."""
        tf = self._make_filter(frames=2, hold=1.0)
        tf.update("moth", 0.90)
        result = tf.update("moth", 0.92)
        assert result["state"] == "CONFIRMED"
        # Acknowledge actuation
        tf.acknowledge_actuation()
        state = tf.get_state()
        assert state in ("HOLD", "ACTUATE")

    def test_history_tracking(self):
        tf = self._make_filter()
        tf.update("grasshopper", 0.91)
        tf.update("beetle", 0.88)
        history = tf.get_history()
        assert len(history) == 2
