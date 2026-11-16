"""EfficientNet-Lite0 feature extractor (timm weights ``efficientnet_lite0.ra_in1k``).

About 3.4 M backbone parameters. No squeeze-excite, ReLU6: the easiest of the three to quantise (INT8).
"""

import timm
from torch import nn

TIMM_NAME = "efficientnet_lite0.ra_in1k"


def build(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return the RGB backbone without classifier and the size of its pooled feature vector.

    The module maps ``[B, 3, H, W]`` to ``[B, features]``.
    """
    model = timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=0)
    return model, model.head_hidden_size
