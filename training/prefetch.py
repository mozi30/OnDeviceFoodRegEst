"""Download a fixed subset of the large datasets once and store it locally (downscaled JPEGs).

Training on the side-angle videos and on MM-Food-100K by streaming is too slow
(hundreds of GB per epoch). This script extracts fixed frames / photos once,
resizes them so that the shorter side is ``--size`` px, and writes

    <out>/<name>/images/<image_id>.jpg
    <out>/<name>/records_<split>.jsonl      one record per image (labels, ``_path``)

``datasets.LocalFoodDataset`` reads them back. Re-running resumes: images that
already exist are not fetched again.

    uv run python -m training.prefetch side --frames 3 --max-video-mb 6
    uv run python -m training.prefetch mmfood --n-train 30000 --n-val 1500
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .datasets import CACHE_DIR, TARGET_KEYS, MMFood100K, Nutrition5kSideAngles

LOCAL_DIR = CACHE_DIR / "local"
log = logging.getLogger("prefetch")


def _save(image: Image.Image, path: Path, size: int) -> None:
    w, h = image.size
    scale = size / min(w, h)
    if scale < 1:
        image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=90)
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def _clean(sample: dict) -> dict:
    return {k: v for k, v in sample.items() if k != "image" and not k.startswith("_")}


def _run(name: str, split: str, records: list, work, threads: int) -> None:
    """``work(rec) -> list[dict]`` (records with ``_path``), run in parallel, appended to the jsonl."""
    out_dir = LOCAL_DIR / name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    done_file = out_dir / f"records_{split}.jsonl"
    done = set()
    if done_file.exists():
        done = {json.loads(line)["_src"] for line in done_file.open()}
    todo = [r for r in records if r["image_id"] not in done]
    log.info("%s/%s: %d sources, %d done, %d to fetch", name, split, len(records), len(done), len(todo))
    lock, t0, n, failed = threading.Lock(), time.time(), 0, 0
    with done_file.open("a") as f, ThreadPoolExecutor(threads) as pool:
        futures = [pool.submit(work, r) for r in todo]
        for fut in as_completed(futures):
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001 - dead links / broken videos are skipped
                failed += 1
                log.debug("failed: %s", e)
                continue
            with lock:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
            if n % 500 == 0:
                rate = n / (time.time() - t0)
                log.info("%s/%s: %d/%d (%.1f/s, eta %.0f min, %d failed)", name, split, n, len(todo), rate,
                         (len(todo) - n) / rate / 60, failed)
    log.info("%s/%s: finished, %d new sources, %d failed", name, split, n, failed)


def side(args) -> None:
    for split in args.splits:
        ds = Nutrition5kSideAngles(split, frames_per_video=args.frames, max_video_mb=args.max_video_mb,
                                   cameras=args.cameras, shuffle=False)
        img_dir = LOCAL_DIR / "nutrition5k_side" / "images"

        def work(rec):
            rows = []
            for s in ds._load(rec):
                path = img_dir / f"{s['image_id']}.jpg"
                _save(s["image"], path, args.size)
                rows.append(_clean(s) | {"dataset": "nutrition5k_side", "_path": str(path), "_src": rec["image_id"]})
            return rows

        _run("nutrition5k_side", split, ds.records, work, args.threads)


def mmfood(args) -> None:
    for split, n in (("val", args.n_val), ("train", args.n_train)):
        ds = MMFood100K(split, min_food_prob=args.min_food_prob)
        usable = [r for r in ds.records if all(math.isfinite(r[k]) and r[k] >= 0 for k in TARGET_KEYS)
                  and 20 <= r["mass_g"] <= 2000]
        random.Random(0).shuffle(usable)
        img_dir = LOCAL_DIR / "mm_food_100k" / "images"

        def work(rec):
            from .datasets._streaming import fetch
            image = Image.open(io.BytesIO(fetch(rec["_url"], timeout=60)))
            image.draft("RGB", (args.size, args.size))
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image).convert("RGB")
            path = img_dir / f"{rec['image_id'].rsplit('.', 1)[0]}.jpg"
            _save(image, path, args.size)
            row = {k: v for k, v in rec.items() if not k.startswith("_")}
            return [row | {"dataset": "mm_food_100k", "_path": str(path), "_src": rec["image_id"]}]

        _run("mm_food_100k", split, usable[:n], work, args.threads)


def convert(args) -> None:
    """Nutrition5k overhead / NutritionVerse-Real: the cached originals -> small local JPEGs (same records)."""
    from .trainer import TrainConfig, build_dataset
    cfg = TrainConfig()
    for name in args.datasets:
        for split in ("train", "val", "test"):
            ds = build_dataset(name, split, cfg, False)
            img_dir = LOCAL_DIR / name / "images"

            def work(rec, ds=ds, img_dir=img_dir, name=name):
                rows = []
                for s in ds._safe_load(rec):
                    path = img_dir / f"{s['image_id']}.jpg"
                    _save(s["image"], path, args.size)
                    rows.append(_clean(s) | {"dataset": name, "_path": str(path), "_src": rec["image_id"]})
                return rows

            _run(name, split, ds.records, work, args.threads)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="what", required=True)
    s = sub.add_parser("side")
    s.add_argument("--frames", type=int, default=3)
    s.add_argument("--max-video-mb", type=int, default=6)
    s.add_argument("--cameras", default="ABCD")
    s.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    m = sub.add_parser("mmfood")
    m.add_argument("--n-train", type=int, default=30000)
    m.add_argument("--n-val", type=int, default=1500)
    m.add_argument("--min-food-prob", type=float, default=0.9)
    c = sub.add_parser("convert")
    c.add_argument("--datasets", nargs="+", default=["nutrition5k", "nutritionverse_real"])
    for q in (s, m, c):
        q.add_argument("--size", type=int, default=320)
        q.add_argument("--threads", type=int, default=24)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("training.datasets._streaming").setLevel(logging.ERROR)
    {"side": side, "mmfood": mmfood, "convert": convert}[args.what](args)


if __name__ == "__main__":
    main()
