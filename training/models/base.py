"""The complete model: feature extractor (+ optional depth input) + ``ClassRegHead``.

Stage 1 (RGB)::

    model = FoodModel("mobilenetv3_large", num_dishes=60, num_ingredients=150)
    out = model(images)                       # images: [B, 3, H, W], ImageNet-normalised
    out["mass_g"], out["kcal"], out["dish_logits"], out["ingredient_logits"], ...

Stage 2 (RGB + depth of Depth Anything, ``depth``: [B, 1, H, W] in about [0, 1])::

    model = FoodModel("mobilenetv3_large", 60, 150, depth="channel")   # or "branch" / "stats"
    model.load_rgb_weights(stage1_checkpoint)
    out = model(images, depth)

The three depth variants (see ``depth.py``) all start out behaving exactly like
the RGB model: every weight that reads depth information is zero at first.
"""

from __future__ import annotations

import torch
from torch import nn

from . import efficientNet_lite0, mobileNetV3_large, mobileNetV4_conv_medium
from .custom_class_reg_head import ClassRegHead
from .depth import DEPTH_MODES, NUM_DEPTH_STATS, DepthBranch, depth_statistics

# name -> function that returns (feature extractor, feature size)
BACKBONES = {
    "efficientnet_lite0": efficientNet_lite0.build,
    "mobilenetv3_large": mobileNetV3_large.build,
    "mobilenetv4_conv_medium": mobileNetV4_conv_medium.build,
}


def _pad_inputs(weight: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Copy ``weight`` into a zero tensor shaped ``like`` that has more input channels / features (dim 1)."""
    out = torch.zeros_like(like)
    out[:, :weight.shape[1]] = weight
    return out


def widen_stem(backbone: nn.Module, in_chans: int) -> None:
    """Give the first convolution ``in_chans`` input channels; RGB filters are kept, new ones are zero."""
    name, old = next((n, m) for n, m in backbone.named_modules() if isinstance(m, nn.Conv2d))
    if old.in_channels == in_chans:
        return
    new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride, old.padding,
                    old.dilation, old.groups, old.bias is not None, old.padding_mode)
    with torch.no_grad():
        new.weight.copy_(_pad_inputs(old.weight, new.weight))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    parent = backbone.get_submodule(name.rpartition(".")[0])
    setattr(parent, name.rpartition(".")[2], new)


class FoodModel(nn.Module):
    """
    Args:
        backbone: one of ``BACKBONES``.
        num_dishes, num_ingredients: sizes of the two recognition outputs of the head (0 = off).
        depth: how depth is used, one of ``DEPTH_MODES``:
            ``none`` (RGB only), ``channel`` (4th input channel of the backbone),
            ``branch`` (separate depth CNN), ``stats`` (hand-made depth statistics).
        pretrained: load ImageNet weights for the backbone.
        depth_features: size of the depth branch's feature vector (``depth="branch"``).
        **head_kwargs: passed to ``ClassRegHead`` (``hidden``, ``dropout``, ``init_mass_g``).
    """

    def __init__(self, backbone: str = "mobilenetv3_large", num_dishes: int = 0, num_ingredients: int = 0,
                 depth: str = "none", pretrained: bool = True, depth_features: int = 128, **head_kwargs):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}, choose from {sorted(BACKBONES)}")
        if depth not in DEPTH_MODES:
            raise ValueError(f"unknown depth mode {depth!r}, choose from {DEPTH_MODES}")
        self.depth = depth
        self.backbone_frozen = False

        self.backbone, features = BACKBONES[backbone](pretrained)
        self.depth_branch = None
        if depth == "channel":
            widen_stem(self.backbone, 4)
        elif depth == "branch":
            self.depth_branch = DepthBranch(depth_features)
            features += depth_features
        elif depth == "stats":
            features += NUM_DEPTH_STATS

        self.head = ClassRegHead(features, num_dishes, num_ingredients, **head_kwargs)
        if depth in ("branch", "stats"):  # the head ignores the depth features until it learns to use them
            first = self.head.shared[1]
            with torch.no_grad():
                first.weight[:, features - (depth_features if depth == "branch" else NUM_DEPTH_STATS):] = 0

    def forward(self, image: torch.Tensor, depth: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.depth == "none":
            return self.head(self.backbone(image))
        if depth is None:
            raise ValueError(f"depth mode {self.depth!r} needs a depth map")
        if self.depth == "channel":
            return self.head(self.backbone(torch.cat([image, depth], dim=1)))
        extra = self.depth_branch(depth) if self.depth == "branch" else depth_statistics(depth)
        return self.head(torch.cat([self.backbone(image), extra], dim=1))

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Train only the head (and the depth branch); the frozen backbone also keeps its BatchNorm statistics."""
        self.backbone_frozen = frozen
        self.backbone.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def load_rgb_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialise this model from a trained RGB ``FoodModel`` (Stage 1 checkpoint).

        Layers that got wider for depth (stem convolution, first head layer) keep
        the RGB weights and zeros for the depth inputs; the depth branch keeps its
        fresh initialisation.
        """
        own = self.state_dict()
        for key, value in state_dict.items():
            if value.shape == own[key].shape:
                own[key] = value
            elif value.dim() >= 2 and value.shape[0] == own[key].shape[0] and value.shape[1] < own[key].shape[1]:
                own[key] = _pad_inputs(value, own[key])
            else:
                raise ValueError(f"shape mismatch for {key}: {tuple(value.shape)} vs {tuple(own[key].shape)}")
        self.load_state_dict(own)
