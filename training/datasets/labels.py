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


@functools.cache
def known_names() -> dict[str, str]:
    """``{name: type}`` of every dish / ingredient name in the map (lower case)."""
    return {name.lower(): kind for (_, _), (kind, name) in label_map().items() if kind in ("dish", "ingredient") and name}


# MM-Food-100K names that are not a visible food
MM_IGNORE = {"water", "ice", "ice cubes"}
# MM-Food ingredient names that mean something else than the map's name ("chili" there is chili con carne)
MM_INGREDIENT_ALIAS = {"chili": "chili peppers", "chilli": "chili peppers", "chilies": "chili peppers"}


def normalize_name(name: str) -> str:
    """Free-text food name -> vocabulary name: lower case, single spaces, and the
    singular / plural form that the map already uses (``tomato`` -> ``tomatoes``)."""
    n = " ".join(name.lower().replace("-", " ").replace("_", " ").split())
    known = known_names()
    if n in known:
        return n
    for cand in (n + "s", n + "es", n[:-1] if n.endswith("s") else None, n[:-2] if n.endswith("es") else None):
        if cand and cand in known:
            return cand
    return n


def mm_food_components(sample: dict) -> tuple[list[str], list[str]]:
    """MM-Food-100K: ``dish_name`` is the dish (or an ingredient when the map knows it as one,
    e.g. "apple"), ``ingredients`` are ingredients unless the map knows them as dishes."""
    known = known_names()
    dishes, ingredients = set(), set()
    names = [(sample.get("dish_name") or "", "dish")] + [(x, "ingredient") for x in sample.get("ingredients") or []]
    for raw, default in names:
        n = normalize_name(raw)
        if default == "ingredient":
            n = MM_INGREDIENT_ALIAS.get(n, n)
        if not n or n in MM_IGNORE:
            continue
        (dishes if known.get(n, default) == "dish" else ingredients).add(n)
    return sorted(dishes), sorted(ingredients)


def components(sample: dict) -> tuple[list[str], list[str]] | None:
    """``(dishes, ingredients)`` of a sample or record in the common vocabulary.

    ``None`` if the sample has no component labels. Labels missing in the map count
    as ingredients under their own name; MM-Food-100K free text is mapped by name
    (``mm_food_components``).
    """
    if sample.get("dataset") == "food101":  # dish label only: add the dish's typical ingredients
        from .food101_ingredients import FOOD101_INGREDIENTS
        return mm_food_components(sample | {"ingredients": FOOD101_INGREDIENTS.get(sample.get("dish_name"), [])})
    if sample.get("dataset") == "mm_food_100k" or "food_type" in sample:
        return mm_food_components(sample)
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
