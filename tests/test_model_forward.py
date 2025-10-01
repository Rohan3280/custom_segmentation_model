import torch
from src.models.vit_backbone import build_vit

def test_model_forward():
    model = build_vit(num_classes=2)
    x = torch.randn(2, 3, 256, 256)  # batch of 2 RGB images
    y = model(x)
    assert y.shape == (2, 2), f"Expected (2,2), got {y.shape}"
