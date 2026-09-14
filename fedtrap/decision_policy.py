"""FedTrap decision policy — maps classification results to actions.

Reads class→action mapping from config.yaml.  The policy is fully
decoupled from the model so that changing crop policy never requires
retraining.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fedtrap.config import get_config, get_class_category, get_class_action


def get_decision(class_name: str, confidence: float,
                 cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """Full decision pipeline: class + confidence → action."""
    cfg = cfg or get_config()
    is_confident = apply_confidence_threshold(confidence, cfg=cfg)
    category = get_class_category(class_name, cfg)
    action = map_class_to_action(class_name, cfg)

    if not is_confident or category == "UNKNOWN":
        return {
            "class_name": class_name,
            "confidence": confidence,
            "category": "UNKNOWN",
            "action": "NO_ACTION",
            "is_confident": False,
        }

    return {
        "class_name": class_name,
        "confidence": confidence,
        "category": category,
        "action": action,
        "is_confident": True,
    }


def classify_prediction(class_name: str, confidence: float,
                        cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """Classify a prediction: apply threshold, return category + action."""
    cfg = cfg or get_config()
    is_confident = apply_confidence_threshold(confidence, cfg=cfg)
    category = get_class_category(class_name, cfg)
    action = map_class_to_action(class_name, cfg)

    if not is_confident or category == "UNKNOWN":
        return {
            "category": "UNKNOWN",
            "action": "NO_ACTION",
            "is_confident": False,
        }

    return {
        "category": category,
        "action": action,
        "is_confident": True,
    }


def map_class_to_action(class_name: str,
                        cfg: Optional[Dict] = None) -> str:
    """Map a class name to its configured action."""
    return get_class_action(class_name, cfg)


def apply_confidence_threshold(confidence: float,
                               threshold: Optional[float] = None,
                               cfg: Optional[Dict] = None) -> bool:
    """Return True if confidence meets the configured threshold."""
    if threshold is None:
        cfg = cfg or get_config()
        decision_cfg = cfg.get("decision", {})
        threshold = decision_cfg.get("confidence_threshold", 0.85)
    return confidence >= threshold
