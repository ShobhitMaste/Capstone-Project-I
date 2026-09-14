"""Tests for decision policy — class→action mapping, thresholds, unknown."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# ---------------------------------------------------------------------------
# Fixtures — minimal config for unit-testing without YAML
# ---------------------------------------------------------------------------
MOCK_CONFIG = {
    "classes": {
        "grasshopper": {"index": 0, "category": "PEST", "action": "CAPTURE"},
        "beetle":      {"index": 1, "category": "PEST", "action": "CAPTURE"},
        "moth":        {"index": 2, "category": "PEST", "action": "CAPTURE"},
        "honeybee":    {"index": 3, "category": "BENEFICIAL", "action": "RELEASE"},
        "butterfly":   {"index": 4, "category": "BENEFICIAL", "action": "RELEASE"},
    },
    "decision": {
        "confidence_threshold": 0.85,
        "confirmation_frames": 3,
        "hold_duration_sec": 2.0,
        "unknown_action": "NO_ACTION",
    },
}


class TestClassToAction:
    """Verify every class maps to the correct action."""

    def test_grasshopper_capture(self):
        from fedtrap.decision_policy import map_class_to_action
        assert map_class_to_action("grasshopper", cfg=MOCK_CONFIG) == "CAPTURE"

    def test_beetle_capture(self):
        from fedtrap.decision_policy import map_class_to_action
        assert map_class_to_action("beetle", cfg=MOCK_CONFIG) == "CAPTURE"

    def test_moth_capture(self):
        from fedtrap.decision_policy import map_class_to_action
        assert map_class_to_action("moth", cfg=MOCK_CONFIG) == "CAPTURE"

    def test_honeybee_release(self):
        from fedtrap.decision_policy import map_class_to_action
        assert map_class_to_action("honeybee", cfg=MOCK_CONFIG) == "RELEASE"

    def test_butterfly_release(self):
        from fedtrap.decision_policy import map_class_to_action
        assert map_class_to_action("butterfly", cfg=MOCK_CONFIG) == "RELEASE"

    def test_unknown_class_no_action(self):
        from fedtrap.decision_policy import map_class_to_action
        assert map_class_to_action("dragonfly", cfg=MOCK_CONFIG) == "NO_ACTION"


class TestConfidenceThreshold:
    """Verify threshold logic."""

    def test_above_threshold_confident(self):
        from fedtrap.decision_policy import apply_confidence_threshold
        assert apply_confidence_threshold(0.90, cfg=MOCK_CONFIG) is True

    def test_exactly_threshold_confident(self):
        from fedtrap.decision_policy import apply_confidence_threshold
        assert apply_confidence_threshold(0.85, cfg=MOCK_CONFIG) is True

    def test_below_threshold_not_confident(self):
        from fedtrap.decision_policy import apply_confidence_threshold
        assert apply_confidence_threshold(0.84, cfg=MOCK_CONFIG) is False

    def test_zero_confidence(self):
        from fedtrap.decision_policy import apply_confidence_threshold
        assert apply_confidence_threshold(0.0, cfg=MOCK_CONFIG) is False


class TestClassifyPrediction:
    """Full decision pipeline."""

    def test_pest_high_confidence(self):
        from fedtrap.decision_policy import classify_prediction
        result = classify_prediction("grasshopper", 0.94, cfg=MOCK_CONFIG)
        assert result["category"] == "PEST"
        assert result["action"] == "CAPTURE"
        assert result["is_confident"] is True

    def test_beneficial_high_confidence(self):
        from fedtrap.decision_policy import classify_prediction
        result = classify_prediction("honeybee", 0.91, cfg=MOCK_CONFIG)
        assert result["category"] == "BENEFICIAL"
        assert result["action"] == "RELEASE"
        assert result["is_confident"] is True

    def test_pest_low_confidence_becomes_unknown(self):
        from fedtrap.decision_policy import classify_prediction
        result = classify_prediction("grasshopper", 0.60, cfg=MOCK_CONFIG)
        assert result["category"] == "UNKNOWN"
        assert result["action"] == "NO_ACTION"
        assert result["is_confident"] is False

    def test_beneficial_low_confidence_becomes_unknown(self):
        from fedtrap.decision_policy import classify_prediction
        result = classify_prediction("butterfly", 0.50, cfg=MOCK_CONFIG)
        assert result["category"] == "UNKNOWN"
        assert result["action"] == "NO_ACTION"
        assert result["is_confident"] is False

    def test_unknown_class_always_no_action(self):
        from fedtrap.decision_policy import classify_prediction
        result = classify_prediction("spider", 0.99, cfg=MOCK_CONFIG)
        assert result["category"] == "UNKNOWN"
        assert result["action"] == "NO_ACTION"
