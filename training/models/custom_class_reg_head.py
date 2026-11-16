"""Head for food recognition and nutrition regression on top of a pooled feature vector.

The head does not regress the macros directly. It predicts

* the amount: ``log(mass in g)`` together with a log-variance (uncertainty range), and
* the composition: protein, fat and carbohydrates per 100 g,

and computes the absolute macros and the energy from both, so that mass, macros
and kcal are consistent by construction (``kcal = 4 P + 4 C + 9 F``).

Two multi-label outputs recognise what is on the plate: dishes (processed /
prepared items such as lasagna or scrambled eggs) and ingredients (plain foods
such as carrot or chicken leg). Both use the common vocabulary of
``datasets/label_map.csv`` and are trained on all datasets.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

KCAL_PER_G = {"protein": 4.0, "fat": 9.0, "carb": 4.0}
MAX_LOG_MASS = math.log(5000.0)  # outputs are bounded to 5 kg (the loss uses the unbounded log mass)


class ClassRegHead(nn.Module):
    """
    Args:
        in_features: size of the backbone's feature vector.
        num_dishes: number of dish labels (0 disables that output).
        num_ingredients: number of ingredient labels (0 disables that output).
        hidden: width of the shared hidden layer.
        dropout: dropout on the features.
        init_mass_g: mass the amount head predicts before training (its bias).
        mode: ``factorized`` (amount x composition, default) or ``direct``: the
            comparison model of the specification that predicts the five totals
            independently (log space); then ``kcal`` is not tied to the macros.

    ``forward`` returns a dict (all tensors ``[B]`` unless noted):

    ``log_mass``, ``log_mass_var``   amount and its uncertainty (in log space)
    ``per_100g``                     ``[B, 3]`` protein, fat, carbohydrates per 100 g
    ``mass_g``, ``kcal``, ``fat_g``, ``carb_g``, ``protein_g``   derived totals
    ``targets``                      ``[B, 5]`` the totals in the order of ``datasets.TARGET_KEYS``
    ``dish_logits``                  ``[B, num_dishes]`` (if enabled)
    ``ingredient_logits``            ``[B, num_ingredients]`` (if enabled)
    """

    def __init__(self, in_features: int, num_dishes: int = 0, num_ingredients: int = 0,
                 hidden: int = 512, dropout: float = 0.2, init_mass_g: float = 200.0,
                 mode: str = "factorized"):
        super().__init__()
        if mode not in ("factorized", "direct"):
            raise ValueError(f"unknown head mode {mode!r}")
        self.mode = mode
        self.shared = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, hidden), nn.ReLU(inplace=True))
        self.amount = nn.Linear(hidden, 2)
        self.composition = nn.Linear(hidden, 3 if mode == "factorized" else 4)  # direct: log kcal, fat, carb, protein
        self.dishes = nn.Linear(hidden, num_dishes) if num_dishes else None
        self.ingredients = nn.Linear(hidden, num_ingredients) if num_ingredients else None

        nn.init.zeros_(self.amount.weight)
        with torch.no_grad():
            self.amount.bias.copy_(torch.tensor([math.log(init_mass_g), 0.0]))
            if mode == "direct":  # start near a typical plate: 250 kcal, 10 g fat, 25 g carbs, 12 g protein
                nn.init.zeros_(self.composition.weight)
                self.composition.bias.copy_(torch.tensor([250.0, 10.0, 25.0, 12.0]).log1p())

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(features)
        log_mass, log_mass_var = self.amount(h).unbind(-1)
        mass_g = log_mass.float().clamp(max=MAX_LOG_MASS).exp()
        if self.mode == "factorized":
            per_100g = F.softplus(self.composition(h))  # >= 0
            protein_g, fat_g, carb_g = (per_100g * mass_g.unsqueeze(-1) / 100).unbind(-1)
            kcal = KCAL_PER_G["protein"] * protein_g + KCAL_PER_G["fat"] * fat_g + KCAL_PER_G["carb"] * carb_g
        else:
            log_totals = self.composition(h)  # log1p of kcal, fat, carb, protein
            kcal, fat_g, carb_g, protein_g = log_totals.float().clamp(max=10).expm1().clamp_min(0).unbind(-1)
            per_100g = torch.stack([protein_g, fat_g, carb_g], -1) / mass_g.unsqueeze(-1) * 100

        out = {
            "log_mass": log_mass, "log_mass_var": log_mass_var, "per_100g": per_100g,
            "mass_g": mass_g, "kcal": kcal, "fat_g": fat_g, "carb_g": carb_g, "protein_g": protein_g,
            "targets": torch.stack([mass_g, kcal, fat_g, carb_g, protein_g], dim=-1),
        }
        if self.mode == "direct":
            out["log_totals"] = log_totals
        if self.dishes is not None:
            out["dish_logits"] = self.dishes(h)
        if self.ingredients is not None:
            out["ingredient_logits"] = self.ingredients(h)
        return out
