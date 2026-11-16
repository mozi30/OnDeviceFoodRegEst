"""Stage 2: the ways a depth map (Depth Anything V2) can enter the model.

The depth map is expected as ``[B, 1, H, W]``, same size as the image and
scaled to about ``[0, 1]`` per image (Depth Anything predicts relative depth).

* ``channel``  depth is stacked to the image as 4th input channel of the backbone
               (nothing needed from this file, see ``base.widen_stem``).
* ``branch``   ``DepthBranch``: a small CNN of its own encodes the depth map into
               a feature vector that is concatenated with the RGB features.
* ``stats``    ``depth_statistics``: fixed, hand-made statistics of the depth map
               are concatenated with the RGB features. No learned depth filters.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

DEPTH_MODES = ("none", "channel", "branch", "stats")
STATS_GRID = 4
NUM_DEPTH_STATS = 4 + STATS_GRID * STATS_GRID


class DepthBranch(nn.Module):
    """Small CNN: ``[B, 1, H, W]`` depth -> ``[B, features]`` (about 0.1 M parameters).

    Four 3x3 convolutions with stride 2 (16x downsampling), global average
    pooling, one linear layer. Trained from scratch.

    Known limitation / idea to try later (not implemented):
        The branch downsamples only 16x and then averages everything, so it
        captures the overall height profile of the scene rather than fine
        spatial detail. The alternative is a second pretrained mobile backbone
        for the depth map (e.g. another MobileNetV3 with a 1-channel stem, its
        1280 features concatenated with the RGB features). That could use depth
        much better, but roughly doubles the on-device cost (latency and model
        size), so it only makes sense if the small branch shows that depth
        helps at all (RQ2) and the latency budget of < 2 s per photo allows it.
    """

    def __init__(self, features: int = 128, widths: tuple[int, ...] = (16, 32, 64, 96)):
        super().__init__()
        layers, c_in = [], 1
        for c_out in widths:
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            c_in = c_out
        self.encoder = nn.Sequential(*layers)
        self.project = nn.Sequential(nn.Linear(c_in, features), nn.ReLU(inplace=True))
        self.features = features

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.project(self.encoder(depth).mean(dim=(2, 3)))


def depth_statistics(depth: torch.Tensor) -> torch.Tensor:
    """``[B, 1, H, W]`` -> ``[B, NUM_DEPTH_STATS]``: mean, std, min, max and the
    mean depth in every cell of a 4x4 grid (relative to the image mean, so the
    grid describes the height profile of the scene)."""
    flat = depth.flatten(1)
    mean = flat.mean(dim=1, keepdim=True)
    grid = F.adaptive_avg_pool2d(depth, STATS_GRID).flatten(1) - mean
    return torch.cat([mean, flat.std(dim=1, keepdim=True), flat.amin(dim=1, keepdim=True),
                      flat.amax(dim=1, keepdim=True), grid], dim=1)
