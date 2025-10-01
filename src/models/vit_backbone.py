import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

def build_vit(num_classes=2, pretrained=False, checkpoint=None):
    model = VisionTransformer(
        image_size=256,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes
    )
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model
