"""
Create a mini version of Vision Transformer based on timm
"""

import torch
from timm.models.vision_transformer import VisionTransformer

def vit_tiny_patch8_32(num_classes=10):
    model = VisionTransformer(img_size=32, 
                patch_size=4, num_classes=num_classes, embed_dim=384, depth=7, num_heads=8, mlp_ratio=1.0)
    return model
