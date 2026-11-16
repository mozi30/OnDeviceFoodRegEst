"""MobileNetV4-Conv-Medium feature extractor (timm weights ``mobilenetv4_conv_medium.e500_r224_in1k``).

About 8.4 M backbone parameters. Pure convolutions (UIB blocks), designed for mobile GPUs and NPUs.
"""

import timm
from torch import nn

TIMM_NAME = "mobilenetv4_conv_medium.e500_r224_in1k"


def build(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return the RGB backbone without classifier and the size of its pooled feature vector.

    The module maps ``[B, 3, H, W]`` to ``[B, features]``.
    """
    model = timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=0)
    return model, model.head_hidden_size
