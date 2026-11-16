"""Datasets prefetched to local disk by ``training.prefetch`` (downscaled JPEGs + records jsonl)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ._streaming import CACHE_DIR, StreamingFoodDataset

LOCAL_DIR = CACHE_DIR / "local"


def local_available(name: str, split: str) -> bool:
    return (LOCAL_DIR / name / f"records_{split}.jsonl").exists()


class LocalFoodDataset(StreamingFoodDataset):
    """Same samples as the streaming class ``name``, read from ``LOCAL_DIR/<name>``.

    Records whose image file is missing are dropped. ``dish_id`` keeps the
    dish split of the source dataset.
    """

    def __init__(self, name: str, split: str, **kwargs):
        kwargs.pop("image_cache_dir", None)
        kwargs.setdefault("prefetch", 0)
        super().__init__(**kwargs)
        self.name, self.split = name, split
        path = LOCAL_DIR / name / f"records_{split}.jsonl"
        seen = set()
        for line in path.open():
            rec = json.loads(line)
            if rec["image_id"] in seen or not Path(rec["_path"]).exists():
                continue
            seen.add(rec["image_id"])
            self.records.append(rec)

    def _load(self, rec: dict) -> dict:
        with Image.open(rec["_path"]) as img:
            return self._sample(rec, img.convert("RGB"))
