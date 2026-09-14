import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import List, Tuple
import numpy as np

def create_mobilenetv2(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = MobileNet_V2_Weights.DEFAULT
    else:
        weights = None
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    return model

def freeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = False

def unfreeze_top_layers(model: nn.Module, n: int = 4) -> None:
    blocks = list(model.features.children())
    num_blocks = len(blocks)
    start_idx = max(0, num_blocks - n)
    for i in range(start_idx, num_blocks):
        for param in blocks[i].parameters():
            param.requires_grad = True

def get_trainable_params(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters() if p.requires_grad]

def set_trainable_params(model: nn.Module, params: List[np.ndarray]) -> None:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) != len(params):
        raise ValueError(f"Expected {len(trainable_params)} params, got {len(params)}")
    with torch.no_grad():
        for p, p_new in zip(trainable_params, params):
            p.copy_(torch.tensor(p_new, dtype=p.dtype, device=p.device))

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)

def load_model(path: str, num_classes: int = 5) -> nn.Module:
    model = create_mobilenetv2(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model
