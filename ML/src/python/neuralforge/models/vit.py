import torch.nn as nn
from ..nn.attention import VisionTransformerBlock

def VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1
):
    return VisionTransformerBlock(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=depth,
        num_classes=num_classes,
        dropout=dropout
    )