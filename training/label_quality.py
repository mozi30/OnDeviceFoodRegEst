"""Sanity rules for nutrition labels (no dependencies, also used by the export environment).

Some early Nutrition5k dishes (mostly February-March 2019, many only in the side-view
videos) carry physically impossible labels, e.g. an olive plate of 7,974 g or
5,966 kcal per 100 g. A single such dish dominates the mean relative error of a whole
test set and gives nonsense targets in training. The rules only reject what cannot be
true; they are not tuned to model errors:

- more than 100 g of protein + fat + carbs per 100 g of food,
- more than 900 kcal per 100 g (pure fat),
- a dish heavier than 3 kg (largest plausible plate in the data: NutritionVerse-Real
  1.7 kg, Nutrition5k 99th percentile 0.7 kg) or lighter than 5 g,
- an empty plate ("plate only") with a weight.
"""

from __future__ import annotations

import math

MAX_MASS_G = 3000.0
MIN_MASS_G = 5.0
MAX_KCAL_PER_100G = 900.0


def label_problem(rec: dict) -> str | None:
    """Why the labels of ``rec`` are impossible, or ``None`` if they are plausible.

    Records without nutrition labels (missing / NaN values) are not judged here.
    """
    try:
        mass = float(rec.get("mass_g"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mass):
        return None
    if mass > MAX_MASS_G:
        return f"mass {mass:.0f} g > {MAX_MASS_G:.0f} g"
    if mass < MIN_MASS_G:
        return f"mass {mass:.0f} g < {MIN_MASS_G:.0f} g"
    names = rec.get("ingredients") or rec.get("food_classes") or []
    if list(names) == ["plate only"]:
        return "empty plate"
    values = {k: rec.get(k) for k in ("kcal", "protein_g", "fat_g", "carb_g")}
    if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()):
        if values["kcal"] / mass * 100 > MAX_KCAL_PER_100G:
            return f"{values['kcal'] / mass * 100:.0f} kcal/100 g > {MAX_KCAL_PER_100G:.0f}"
        macros = (values["protein_g"] + values["fat_g"] + values["carb_g"]) / mass * 100
        if macros > 100:
            return f"{macros:.0f} g macros/100 g > 100"
    return None


def clean(records: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Split ``records`` into plausible ones and ``(record, reason)`` of the rejected ones."""
    kept, dropped = [], []
    for r in records:
        reason = label_problem(r)
        if reason is None:
            kept.append(r)
        else:
            dropped.append((r, reason))
    return kept, dropped
