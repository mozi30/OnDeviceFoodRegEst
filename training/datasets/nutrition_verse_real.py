"""NutritionVerse-Real, streamed out of the Kaggle archive with HTTP range requests.

Source: https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real
(CC BY-NC-SA 4.0). Kaggle serves the dataset as one 1.1 GB zip without login;
the zip's directory is read once (3 small requests) and every photo is then
fetched with a single range request, so the archive is never downloaded.
"""

from __future__ import annotations

import csv
import io
import json
import re

from ._streaming import CACHE_DIR, RemoteZip, StreamingFoodDataset, hash_split, session, to_float, _write_atomic

KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/nutritionverse/nutritionverse-real"
DISH_CSV = "nutritionverse_dish_metadata3.csv"
ROOT = "nutritionverse-manual/nutritionverse-manual"
SPLIT_CSV = f"{ROOT}/updated-manual-dataset-splits.csv"
MAX_ITEMS = 7


def _resolve_url() -> str:
    """Kaggle answers with a redirect to a signed storage URL (valid for 3 days)."""
    r = session().get(KAGGLE_DOWNLOAD, allow_redirects=False, timeout=60)
    if r.status_code not in (301, 302, 303, 307) or "Location" not in r.headers:
        raise RuntimeError(f"Kaggle did not hand out a download URL (HTTP {r.status_code}); "
                           "anonymous download may have been disabled")
    return r.headers["Location"]


def _build_index(archive: RemoteZip) -> dict:
    members = archive.index()

    def text(name: str) -> str:
        return archive.read(name, members[name]).decode("utf-8-sig")

    dishes = {}
    for row in csv.DictReader(io.StringIO(text(DISH_CSV))):
        items = [(re.sub(r"-\d+$", "", row[f"food_item_type_{i}"]), to_float(row[f"food_weight_g_{i}"]))
                 for i in range(1, MAX_ITEMS + 1) if row.get(f"food_item_type_{i}")]
        dishes[f"dish_{row['dish_id']}"] = {
            "mass_g": to_float(row["total_food_weight"]),
            "kcal": to_float(row["total_calories"]),
            "fat_g": to_float(row["total_fats"]),
            "carb_g": to_float(row["total_carbohydrates"]),
            "protein_g": to_float(row["total_protein"]),
            "food_classes": [name for name, _ in items],
            "food_class_grams": [grams for _, grams in items],
        }
    official = {r["file_name"]: r["category"].lower() for r in csv.DictReader(io.StringIO(text(SPLIT_CSV)))}
    images = {name: entry for name, entry in members.items() if name.lower().endswith((".jpg", ".jpeg", ".png"))}
    return {"size": archive.size, "images": images, "dishes": dishes, "official": official}


class NutritionVerseReal(StreamingFoodDataset):
    """889 iPhone photos (4032x3024) of 225 weighed dishes with food item types.

    Args:
        split: with ``split_source="official"`` ``train`` / ``val`` / ``all``
            (the authors' split, which is by dish); with ``split_source="hash"``
            ``train`` / ``val`` / ``test`` / ``all``, assigned by dish id.
        split_source: ``official`` or ``hash``.
        fractions: train/val/test shares for ``split_source="hash"``.
        **kwargs: see ``StreamingFoodDataset``. ``draft_size`` is recommended,
            the photos are 12 MP.

    Extra sample keys: ``food_classes`` (names), ``food_class_grams``.
    ``self.classes`` lists all food item types of the whole dataset.
    """

    name = "nutritionverse_real"

    def __init__(self, split: str = "train", split_source: str = "official",
                 fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), **kwargs):
        super().__init__(**kwargs)
        splits = {"official": ("train", "val"), "hash": ("train", "val", "test")}
        if split_source not in splits:
            raise ValueError(f"unknown split_source {split_source!r}")
        if split != "all" and split not in splits[split_source]:
            raise ValueError(f"split must be 'all' or one of {splits[split_source]} for split_source={split_source!r}")
        self.split = split
        self._archive = RemoteZip(_resolve_url)

        cache = CACHE_DIR / self.name / "index.json"
        index = json.loads(cache.read_text()) if cache.exists() else None
        if index is None or index["size"] != self._archive.size:  # archive changed -> offsets invalid
            index = _build_index(self._archive)
            _write_atomic(cache, json.dumps(index).encode())

        for path, entry in sorted(index["images"].items()):
            file_name = path.rsplit("/", 1)[1]
            dish_id = file_name.split("_IMG")[0]
            if dish_id not in index["dishes"]:
                continue
            if split_source == "official":
                part = index["official"].get(file_name)
            else:
                part = splits["hash"][hash_split(dish_id, fractions)]
            if split in ("all", part):
                self.records.append({"dish_id": dish_id, "image_id": file_name.split(".rf.")[0],
                                     **index["dishes"][dish_id], "_path": path, "_entry": entry})
        self.classes = sorted({c for d in index["dishes"].values() for c in d["food_classes"]})

    def _load(self, rec: dict) -> dict:
        image = self._image(f"{rec['image_id']}.jpg", lambda: self._archive.read(rec["_path"], rec["_entry"]))
        return self._sample(rec, image)
