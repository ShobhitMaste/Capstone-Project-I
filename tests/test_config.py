"""Tests for YAML configuration loader."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestConfigLoader:
    """Verify config loading and typed accessors."""

    def test_load_config_returns_dict(self):
        from fedtrap.config import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_classes_present(self):
        from fedtrap.config import load_config
        cfg = load_config()
        if "classes" in cfg:
            assert "grasshopper" in cfg["classes"]
            assert "beetle" in cfg["classes"]
            assert "moth" in cfg["classes"]
            assert "honeybee" in cfg["classes"]
            assert "butterfly" in cfg["classes"]

    def test_get_class_names_ordered(self):
        from fedtrap.config import get_class_names
        names = get_class_names()
        assert len(names) == 5
        assert names[0] == "grasshopper"
        assert names[4] == "butterfly"

    def test_get_class_to_index(self):
        from fedtrap.config import get_class_to_index
        c2i = get_class_to_index()
        assert c2i["grasshopper"] == 0
        assert c2i["butterfly"] == 4

    def test_get_index_to_class(self):
        from fedtrap.config import get_index_to_class
        i2c = get_index_to_class()
        assert i2c[0] == "grasshopper"
        assert i2c[3] == "honeybee"

    def test_get_class_action(self):
        from fedtrap.config import get_class_action
        assert get_class_action("grasshopper") == "CAPTURE"
        assert get_class_action("honeybee") == "RELEASE"
        assert get_class_action("unknown_insect") == "NO_ACTION"

    def test_get_class_category(self):
        from fedtrap.config import get_class_category
        assert get_class_category("grasshopper") == "PEST"
        assert get_class_category("butterfly") == "BENEFICIAL"
        assert get_class_category("spider") == "UNKNOWN"

    def test_defaults_for_missing_keys(self):
        from fedtrap.config import _deep_merge
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        merged = _deep_merge(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3

    def test_project_root(self):
        from fedtrap.config import get_project_root
        root = get_project_root()
        assert root.is_dir()
