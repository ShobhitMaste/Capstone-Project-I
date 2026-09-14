"""FedTrap configuration loader.

Reads config/config.yaml and exposes typed helpers so no other module
has to parse YAML or hard-code class names / thresholds / GPIO pins.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Defaults (used when a key is missing from YAML)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "decision": {
        "confidence_threshold": 0.85,
        "confirmation_frames": 3,
        "hold_duration_sec": 2.0,
        "unknown_action": "NO_ACTION",
    },
    "servo": {
        "gpio": 18,
        "home_angle": 90,
        "capture_angle": 150,
        "release_angle": 30,
        "pwm_frequency": 50,
        "actuation_timeout_sec": 5.0,
        "use_pigpio": True,
    },
    "model": {
        "architecture": "mobilenetv2",
        "num_classes": 5,
        "input_size": 224,
        "pretrained": True,
        "weights_path": "models/classifier_best.pt",
    },
    "training": {
        "epochs": 15,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seed": 42,
        "img_size": 224,
        "freeze_epochs": 5,
        "num_workers": 2,
    },
    "federated": {
        "server_address": "0.0.0.0:8080",
        "rounds": 10,
        "local_epochs": 1,
        "min_clients": 2,
        "method": "fedavg",
        "fedprox_mu": 0.01,
        "trainable_layers": "classifier",
    },
    "trap": {"id": "trap_a", "role": "server_client"},
}

# ---------------------------------------------------------------------------
# Project-root resolver
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """Return the project root (directory that contains config/)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    # Walk up from this file to find the directory containing config/
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "config").is_dir():
        _PROJECT_ROOT = candidate
        return _PROJECT_ROOT
    # Fallback: cwd
    _PROJECT_ROOT = Path.cwd()
    return _PROJECT_ROOT


def set_project_root(path: Path) -> None:
    global _PROJECT_ROOT
    _PROJECT_ROOT = Path(path).resolve()


# ---------------------------------------------------------------------------
# Config singleton
# ---------------------------------------------------------------------------
_CONFIG: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config, merge with defaults, cache globally."""
    global _CONFIG
    if _CONFIG is not None and config_path is None:
        return _CONFIG

    if config_path is None:
        config_path = str(get_project_root() / "config" / "config.yaml")

    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    else:
        user_cfg = {}

    # Deep merge defaults ← user
    cfg = _deep_merge(_DEFAULTS, user_cfg)
    _CONFIG = cfg
    return cfg


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return the cached config (loads from path if needed)."""
    if _CONFIG is None or config_path is not None:
        return load_config(config_path)
    return _CONFIG


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------

def get_class_names(cfg: Optional[Dict] = None) -> List[str]:
    """Ordered list of class names (by index)."""
    cfg = cfg or get_config()
    classes = cfg.get("classes", {})
    ordered = sorted(classes.items(), key=lambda kv: kv[1].get("index", 0))
    return [name for name, _ in ordered]


def get_class_to_index(cfg: Optional[Dict] = None) -> Dict[str, int]:
    cfg = cfg or get_config()
    classes = cfg.get("classes", {})
    return {name: info["index"] for name, info in classes.items()}


def get_index_to_class(cfg: Optional[Dict] = None) -> Dict[int, str]:
    cfg = cfg or get_config()
    classes = cfg.get("classes", {})
    return {info["index"]: name for name, info in classes.items()}


def get_class_action(class_name: str, cfg: Optional[Dict] = None) -> str:
    cfg = cfg or get_config()
    classes = cfg.get("classes", {})
    info = classes.get(class_name, {})
    return info.get("action", "NO_ACTION")


def get_class_category(class_name: str, cfg: Optional[Dict] = None) -> str:
    cfg = cfg or get_config()
    classes = cfg.get("classes", {})
    info = classes.get(class_name, {})
    return info.get("category", "UNKNOWN")


def load_labels_json(path: Optional[str] = None) -> Dict[int, str]:
    """Load labels.json → {int_index: class_name}."""
    if path is None:
        path = str(get_project_root() / "labels.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_labels_json(mapping: Dict[int, str], path: Optional[str] = None) -> None:
    if path is None:
        path = str(get_project_root() / "labels.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(mapping.items())}, f, indent=4)
