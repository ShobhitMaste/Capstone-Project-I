"""FedTrap temporal confirmation filter — multi-frame state machine.

States: IDLE → CANDIDATE → CONFIRMED → ACTUATE → HOLD → RESET

Requires N consecutive same-class predictions above the confidence
threshold before confirming.  Mixed or low-confidence predictions
reset the counter.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class TemporalFilter:
    """Thread-safe FSM for temporal prediction confirmation."""

    STATES = ("IDLE", "CANDIDATE", "CONFIRMED", "ACTUATE", "HOLD", "RESET")

    def __init__(
        self,
        confirmation_frames: int = 3,
        hold_duration: float = 2.0,
        confidence_threshold: float = 0.85,
    ):
        self.confirmation_frames = max(1, confirmation_frames)
        self.hold_duration = hold_duration
        self.confidence_threshold = confidence_threshold

        self._state: str = "IDLE"
        self._candidate: Optional[str] = None
        self._consecutive: int = 0
        self._hold_start: float = 0.0
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        class_name: str,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Feed a new prediction; returns current state + class_name."""
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            self._history.append(
                {"class_name": class_name, "confidence": confidence, "timestamp": ts}
            )
            if len(self._history) > 200:
                self._history = self._history[-100:]

            # --- HOLD: wait until duration expires ---
            if self._state == "HOLD":
                if ts - self._hold_start >= self.hold_duration:
                    self._reset_internal()
                return self._result()

            # --- Confidence gate ---
            if confidence < self.confidence_threshold:
                self._reset_internal()
                return self._result()

            # --- Consistency check ---
            if self._candidate == class_name:
                self._consecutive += 1
            else:
                self._candidate = class_name
                self._consecutive = 1

            # --- State advancement ---
            if self._consecutive >= self.confirmation_frames:
                self._state = "CONFIRMED"
            elif self._consecutive > 1:
                self._state = "CANDIDATE"
            else:
                self._state = "IDLE" if self.confirmation_frames > 1 else "CONFIRMED"

            return self._result()

    def acknowledge_actuation(self) -> None:
        """Called after servo has actuated; transitions to HOLD."""
        with self._lock:
            self._state = "HOLD"
            self._hold_start = time.time()

    def reset(self) -> None:
        with self._lock:
            self._reset_internal()

    def get_state(self) -> str:
        with self._lock:
            return self._state

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_internal(self) -> None:
        self._state = "IDLE"
        self._candidate = None
        self._consecutive = 0

    def _result(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "class_name": self._candidate,
            "consecutive": self._consecutive,
        }
