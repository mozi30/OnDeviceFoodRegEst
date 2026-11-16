"""MobileNetV3-Large (1.0) feature extractor (timm weights ``mobilenetv3_large_100.ra_in1k``).

About 4.2 M backbone parameters. Uses squeeze-excite and hard-swish.
"""

import timm
from torch import nn

TIMM_NAME = "mobilenetv3_large_100.ra_in1k"


def build(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return the RGB backbone without classifier and the size of its pooled feature vector.

    The module maps ``[B, 3, H, W]`` to ``[B, features]``.
    """
    model = timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=0)
    return model, model.head_hidden_size
