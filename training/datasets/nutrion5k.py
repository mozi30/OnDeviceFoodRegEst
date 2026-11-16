"""Nutrition5k, streamed from the official public Google Cloud bucket.

Source: https://storage.googleapis.com/nutrition5k_dataset (CC BY 4.0).
Nothing of the 181 GB set is downloaded up front. Two views are available:

* ``Nutrition5k``: the overhead RealSense frames (``rgb.png`` ~0.4 MB, optionally
  ``depth_raw.png``), one request per image.
* ``Nutrition5kSideAngles``: frames of the side-angle videos. A video (raw
  ``.h264``, typically 2-20 MB) is fetched, a few frames at different time
  positions are decoded in memory and the video is thrown away.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re

import av
import numpy as np
from PIL import Image

from ._streaming import CACHE_DIR, StreamingFoodDataset, cached, fetch, hash_split, session, _write_atomic

BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
LIST_API = "https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o"
DEPTH_UNITS_PER_M = 10_000  # depth_raw.png is 16 bit, 1 unit = 0.1 mm


def _meta(path: str) -> str:
    return cached(CACHE_DIR / "nutrition5k" / path, lambda: fetch(f"{BASE}/{path}")).decode()


def _overhead_dishes() -> set[str]:
    """Dish ids that have overhead imagery (3,490 of the 5,006 dishes)."""
    def produce() -> bytes:
        prefix, dishes, token = "nutrition5k_dataset/imagery/realsense_overhead/", [], None
        while True:
            params = {"prefix": prefix, "delimiter": "/", "maxResults": 5000,
                      "fields": "prefixes,nextPageToken"}
            if token:
                params["pageToken"] = token
            r = session().get(LIST_API, params=params, timeout=60)
            r.raise_for_status()
            page = r.json()
            dishes += [p[len(prefix):].strip("/") for p in page.get("prefixes", [])]
            if not (token := page.get("nextPageToken")):
                return json.dumps(dishes).encode()

    return set(json.loads(cached(CACHE_DIR / "nutrition5k" / "overhead_dishes.json", produce)))


def _side_videos() -> dict[str, dict[str, int]]:
    """``{dish id: {camera letter: video size in bytes}}`` for all side-angle videos."""
    def produce() -> bytes:
        prefix, videos, token = "nutrition5k_dataset/imagery/side_angles/", {}, None
        while True:
            params = {"prefix": prefix, "maxResults": 5000, "fields": "items(name,size),nextPageToken"}
            if token:
                params["pageToken"] = token
            r = session().get(LIST_API, params=params, timeout=60)
            r.raise_for_status()
            page = r.json()
            for item in page.get("items", []):
                if m := re.fullmatch(r"(dish_\d+)/camera_([A-D])\.h264", item["name"][len(prefix):]):
                    videos.setdefault(m.group(1), {})[m.group(2)] = int(item["size"])
            if not (token := page.get("nextPageToken")):
                return json.dumps(videos).encode()

    return json.loads(cached(CACHE_DIR / "nutrition5k" / "side_videos.json", produce))


_START_CODE = re.compile(b"\x00\x00\x01")


def _frame_offsets(data: bytes) -> tuple[list[int], list[int]]:
    """Byte offset of every frame in a raw H.264 stream, and which frames are keyframes."""
    offsets, keyframes = [], []
    for m in _START_CODE.finditer(data):
        o = m.start()
        if o + 4 >= len(data):
            break
        nal_type = data[o + 3] & 31
        if nal_type in (1, 5) and data[o + 4] & 0x80:  # slice with first_mb_in_slice == 0
            if nal_type == 5:
                keyframes.append(len(offsets))
            offsets.append(o)
    return offsets, keyframes


def _decode_frames(data: bytes, indices: list[int], offsets: list[int], keyframes: list[int]) -> dict[int, Image.Image]:
    """Decode the frames ``indices``; decoding starts at the keyframe before each one."""
    header = data[:offsets[0]]  # SPS / PPS
    by_gop: dict[int, list[int]] = {}
    for i in indices:
        by_gop.setdefault(max(k for k in keyframes if k <= i), []).append(i)
    out: dict[int, Image.Image] = {}
    for key, wanted in by_gop.items():
        following = [k for k in keyframes if k > key]
        end = offsets[following[0]] if following else len(data)
        ctx = av.CodecContext.create("h264", "r")
        packets = ctx.parse(header + data[offsets[key]:end]) + ctx.parse() + [None]  # None flushes the decoder
        n = key
        for packet in packets:
            for frame in ctx.decode(packet):
                if n in wanted:
                    out[n] = frame.to_image()
                n += 1
            if n > max(wanted):
                break
    return out


def _dishes() -> dict[str, dict]:
    """Parse the dish metadata: totals, then 7 fields per ingredient."""
    dishes = {}
    for cafe in ("cafe1", "cafe2"):
        for row in csv.reader(io.StringIO(_meta(f"metadata/dish_metadata_{cafe}.csv"))):
            ingr = [row[i:i + 7] for i in range(6, len(row) - 6, 7)]
            dishes[row[0]] = {
                "dish_id": row[0],
                "image_id": row[0],
                "kcal": float(row[1]),
                "mass_g": float(row[2]),
                "fat_g": float(row[3]),
                "carb_g": float(row[4]),
                "protein_g": float(row[5]),
                "ingredients": [x[1] for x in ingr],
                "ingredient_grams": [float(x[2]) for x in ingr],
            }
    return dishes


class Nutrition5k(StreamingFoodDataset):
    """Overhead RGB(-D) frames of Nutrition5k with dish totals and ingredient lists.

    Args:
        split: ``train`` / ``val`` / ``test`` / ``all``. ``test`` is the official
            test split; ``val`` is carved out of the official train split by dish.
        depth: also load the raw depth map as ``sample["depth"]`` (float32 metres,
            0 = no reading, not transformed) and use the official depth splits.
        val_fraction: share of the official train dishes used for ``val``.
        **kwargs: see ``StreamingFoodDataset``.

    Extra sample keys: ``ingredients`` (names), ``ingredient_grams``.
    Dishes with a non-positive total mass (broken labels) are dropped.
    """

    name = "nutrition5k"

    def __init__(self, split: str = "train", depth: bool = False, val_fraction: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"unknown split {split!r}")
        self.split, self.depth = split, depth

        kind = "depth" if depth else "rgb"
        train = _meta(f"dish_ids/splits/{kind}_train_ids.txt").split()
        test = _meta(f"dish_ids/splits/{kind}_test_ids.txt").split()
        if split == "test":
            ids = test
        elif split == "all":
            ids = train + test
        else:
            part = 1 if split == "val" else 0
            ids = [d for d in train if hash_split(d, (1 - val_fraction, val_fraction)) == part]

        dishes, available = _dishes(), _overhead_dishes()
        self.records = [dishes[d] for d in sorted(set(ids))
                        if d in dishes and d in available and dishes[d]["mass_g"] > 0]
        self.ingredient_names = sorted({n for r in self.records for n in r["ingredients"]})

    def _load(self, rec: dict) -> dict:
        folder = f"imagery/realsense_overhead/{rec['dish_id']}"
        image = self._image(f"{rec['dish_id']}_rgb.png", lambda: fetch(f"{BASE}/{folder}/rgb.png"))
        extra = {}
        if self.depth:
            raw = self._bytes(f"{rec['dish_id']}_depth_raw.png", lambda: fetch(f"{BASE}/{folder}/depth_raw.png"))
            extra["depth"] = np.asarray(Image.open(io.BytesIO(raw)), dtype=np.float32) / DEPTH_UNITS_PER_M
        return self._sample(rec, image, **extra)


class Nutrition5kSideAngles(StreamingFoodDataset):
    """Frames of the Nutrition5k side-angle videos (1920x1080, cameras A-D).

    One record is one video; it yields ``frames_per_video`` samples taken at
    evenly spaced time positions. With ``shuffle=True`` the position inside each
    time segment is random and changes every epoch, and the frames of different
    videos are mixed through a shuffle buffer. ``len()`` counts videos.

    Args:
        split: ``train`` / ``val`` / ``test`` / ``all``, same dish split as ``Nutrition5k``.
        frames_per_video: frames taken from every video.
        cameras: which of the cameras ``A``-``D`` to use.
        max_video_mb: only the first ``max_video_mb`` MB of a video are fetched
            (a handful of recordings are hundreds of MB long).
        val_fraction: share of the official train dishes used for ``val``.
        **kwargs: see ``StreamingFoodDataset``. With ``image_cache_dir`` the
            extracted frames (not the videos) are stored as JPEG; they are reused
            when the frame positions are fixed, i.e. without ``shuffle``.

    Extra sample keys: ``camera``, ``frame_index``, ``ingredients``,
    ``ingredient_grams``.
    """

    name = "nutrition5k_side"

    def __init__(self, split: str = "train", frames_per_video: int = 4, cameras: str = "ABCD",
                 max_video_mb: int = 48, val_fraction: float = 0.1, **kwargs):
        kwargs.setdefault("shuffle_buffer", 16 * frames_per_video)
        super().__init__(**kwargs)
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"unknown split {split!r}")
        self.split, self.frames_per_video, self.max_bytes = split, frames_per_video, max_video_mb * 2**20

        train = _meta("dish_ids/splits/rgb_train_ids.txt").split()
        test = _meta("dish_ids/splits/rgb_test_ids.txt").split()
        if split == "test":
            ids = test
        elif split == "all":
            ids = train + test
        else:
            part = 1 if split == "val" else 0
            ids = [d for d in train if hash_split(d, (1 - val_fraction, val_fraction)) == part]

        dishes, videos = _dishes(), _side_videos()
        for d in sorted(set(ids)):
            if d not in dishes or dishes[d]["mass_g"] <= 0:
                continue
            for cam, size in sorted(videos.get(d, {}).items()):
                if cam in cameras and size > 100_000:  # a few videos are empty
                    self.records.append({**dishes[d], "image_id": f"{d}_camera_{cam}", "camera": cam, "_size": size})
        self.ingredient_names = sorted({n for r in self.records for n in r["ingredients"]})

    def _positions(self, rec: dict, frame_count: int) -> list[int]:
        n = self.frames_per_video
        rng = random.Random(f"{self.seed}-{self.epoch}-{rec['image_id']}") if self.shuffle else None
        fractions = [(k + (rng.random() if rng else 0.5)) / n for k in range(n)]
        return sorted({min(frame_count - 1, int(f * frame_count)) for f in fractions})

    def _load(self, rec: dict) -> list[dict]:
        n = self.frames_per_video
        cache = None if self.shuffle or self.image_cache_dir is None else self.image_cache_dir / self.name
        if cache is not None:
            paths = sorted(cache.glob(f"{rec['image_id']}_{n}frames_f*.jpg"))
            if len(paths) == n:
                return [self._sample(rec, Image.open(p).convert("RGB"), image_id=p.stem.replace(f"_{n}frames", ""),
                                     frame_index=int(p.stem[-4:])) for p in paths]

        url = f"{BASE}/imagery/side_angles/{rec['dish_id']}/camera_{rec['camera']}.h264"
        headers = {"Range": f"bytes=0-{self.max_bytes - 1}"} if rec["_size"] > self.max_bytes else None
        data = fetch(url, headers=headers, timeout=180)
        offsets, keyframes = _frame_offsets(data)
        if rec["_size"] > self.max_bytes:
            offsets = offsets[:-1]  # the last frame is cut off
        if not offsets or not keyframes:
            raise OSError("no decodable frames")
        frames = _decode_frames(data, self._positions(rec, len(offsets)), offsets, keyframes)

        samples = []
        for index, image in sorted(frames.items()):
            if cache is not None and len(frames) == n:
                buf = io.BytesIO()
                image.save(buf, "JPEG", quality=92)
                _write_atomic(cache / f"{rec['image_id']}_{n}frames_f{index:04d}.jpg", buf.getvalue())
            samples.append(self._sample(rec, image, image_id=f"{rec['image_id']}_f{index:04d}",
                                        frame_index=index))
        return samples
