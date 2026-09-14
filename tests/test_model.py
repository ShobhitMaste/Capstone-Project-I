"""Tests for MobileNetV2 model builder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch


class TestModelCreation:
    """Verify model structure and parameter management."""

    def test_create_model_default(self):
        from fedtrap.model import create_mobilenetv2
        model = create_mobilenetv2(num_classes=5, pretrained=False)
        assert model is not None

    def test_output_shape(self):
        from fedtrap.model import create_mobilenetv2
        model = create_mobilenetv2(num_classes=5, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_five_class_output(self):
        from fedtrap.model import create_mobilenetv2
        model = create_mobilenetv2(num_classes=5, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_freeze_backbone(self):
        from fedtrap.model import create_mobilenetv2, freeze_backbone
        model = create_mobilenetv2(num_classes=5, pretrained=False)
        freeze_backbone(model)
        # All feature params should be frozen
        for name, param in model.features.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"
        # Classifier should still be trainable
        for name, param in model.classifier.named_parameters():
            assert param.requires_grad, f"classifier {name} should be trainable"

    def test_count_parameters(self):
        from fedtrap.model import create_mobilenetv2, count_parameters
        model = create_mobilenetv2(num_classes=5, pretrained=False)
        total, trainable = count_parameters(model)
        assert total > 0
        assert trainable > 0
        assert trainable <= total

    def test_get_set_trainable_params(self):
        from fedtrap.model import create_mobilenetv2, get_trainable_params, set_trainable_params
        model = create_mobilenetv2(num_classes=5, pretrained=False)
        params = get_trainable_params(model)
        assert isinstance(params, list)
        assert len(params) > 0
        # Set them back (should not raise)
        set_trainable_params(model, params)

    def test_save_load_model(self, tmp_path):
        from fedtrap.model import create_mobilenetv2, save_model, load_model
        model = create_mobilenetv2(num_classes=5, pretrained=False)
        path = str(tmp_path / "test_model.pt")
        save_model(model, path)
        loaded = load_model(path, num_classes=5)
        # Verify outputs match
        model.eval()
        loaded.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = loaded(x)
        assert torch.allclose(out1, out2, atol=1e-5)
