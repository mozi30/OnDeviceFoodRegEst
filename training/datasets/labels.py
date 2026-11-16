"""Common label vocabulary of the datasets: every component of a dish is a *dish* or an *ingredient*.

``label_map.csv`` (edit it by hand) lists every label of Nutrition5k ("ingredients")
and NutritionVerse-Real ("food item types") with

    type   ``ingredient``  a plain food (carrot, chicken leg, white rice, olive oil)
           ``dish``        a processed / prepared item (scrambled eggs, french fries, lasagna)
           ``ignore``      not a food (e.g. "plate only")
    name   the common name; labels of both datasets with the same name are the same class

Rule used for the first version: a label is a dish when its name states a
preparation or it is a composed item; a single food without a stated
preparation is an ingredient. Distinctions such as chicken leg / chicken breast
or scrambled eggs / boiled egg / eggs are kept.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

MAP_FILE = Path(__file__).with_name("label_map.csv")


@functools.cache
def label_map() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(dataset, label): (type, name)}``"""
    with MAP_FILE.open(newline="") as f:
        return {(r["dataset"], r["label"]): (r["type"], r["name"]) for r in csv.DictReader(f)}


def components(sample: dict) -> tuple[list[str], list[str]] | None:
    """``(dishes, ingredients)`` of a sample or record in the common vocabulary.

    ``None`` if the dataset has no usable component labels (MM-Food-100K: free text).
    Labels missing in the map count as ingredients under their own name.
    """
    if "food_classes" in sample:
        dataset, labels = "nutritionverse_real", sample["food_classes"]
    elif "ingredient_grams" in sample:
        dataset, labels = "nutrition5k", sample["ingredients"]
    else:
        return None
    dishes, ingredients = set(), set()
    for label in labels:
        kind, name = label_map().get((dataset, label), ("ingredient", label.replace("-", " ")))
        if kind == "dish":
            dishes.add(name)
        elif kind == "ingredient":
            ingredients.add(name)
    return sorted(dishes), sorted(ingredients)
