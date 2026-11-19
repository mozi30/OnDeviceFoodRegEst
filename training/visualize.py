"""Click through training samples exactly as a training run sees them (local web viewer).

Replays the sample order of a run without loading any image up front: per-epoch
resampling (``--sampling``), per-dataset shuffling, the split across DataLoader
workers, the size-weighted mixing of the datasets (``Interleave``) and the batches.
For each sample it shows

- the stored photo and several random draws of the training transform (the model
  input, before normalisation),
- the targets the loss uses (mass, log mass, composition per 100 g, totals),
- the multi-hot labels of the two recognition heads after mapping to the run's
  vocabulary (unknown names count as "other"),
- optionally the prediction of a checkpoint.

Val / test splits of each dataset can be browsed too (eval transform, stored order).

    uv run python -m training.visualize --run runs/final_hires/mobilenetv3_large_img320_s0
    uv run python -m training.visualize --run runs/final_hires/mobilenetv3_large_img320_s0 --predict
    uv run python -m training.visualize --datasets nutrition5k nutrition5k_side --sampling nutrition5k_side=0.25

then open http://127.0.0.1:8765 (arrow keys: previous / next sample, A: new augmentation, R: random).
The augmentation draws are random like in training; everything else is reproduced exactly.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import math
import random
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from .datasets import TARGET_KEYS
from .datasets.labels import components
from .trainer import (TrainConfig, build_dataset, build_transform, parse_sampling, primary_label, resample)


# ------------------------------------------------------------------ replay --

class TrainingStream:
    """Record-level replay of the training sample order (``trainer.train`` + DataLoader)."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.sampling = parse_sampling(cfg)
        self.sets = {n: build_dataset(n, "train", cfg, True) for n in cfg.datasets}
        for ds in self.sets.values():
            ds.transform = None
        self._cache: dict[int, list[list[tuple[str, dict]]]] = {}

    def epoch(self, epoch: int) -> list[list[tuple[str, dict]]]:
        """Batches of ``(dataset, record)`` for one epoch, in the order the training loop gets them."""
        if epoch in self._cache:
            return self._cache[epoch]
        # resampling uses one rng over all epochs (trainer: rng = Random(seed), resample every epoch)
        rng = random.Random(self.cfg.seed)
        records = {n: list(ds.records) for n, ds in self.sets.items()}
        for e in range(1, epoch + 1):
            for name, factor in self.sampling.items():
                if name in self.sets:
                    ds = self.sets[name]
                    resample(ds, factor, rng)
                    records[name] = list(ds.records)
        for name, ds in self.sets.items():  # leave the datasets as they were
            ds.records = list(getattr(ds, "all_records", ds.records))

        workers = max(1, self.cfg.workers)
        per_worker = []
        for w in range(workers):
            streams = []
            for name in self.cfg.datasets:
                recs = list(records[name])
                random.Random(f"{self.cfg.seed}-{epoch}").shuffle(recs)  # StreamingFoodDataset._stream
                streams.append([name, iter(recs[w::workers]), len(records[name])])
            mix = random.Random(f"{self.cfg.seed}-{epoch}-{w}-mix")       # Interleave.__iter__
            order = []
            streams = [s for s in streams if s[2]]
            while streams:
                i = mix.choices(range(len(streams)), weights=[s[2] for s in streams])[0]
                try:
                    order.append((streams[i][0], next(streams[i][1])))
                except StopIteration:
                    del streams[i]
            bs = self.cfg.batch_size
            per_worker.append([order[k:k + bs] for k in range(0, len(order) - bs + 1, bs)])  # drop_last
        batches, k = [], 0  # DataLoader takes batches from the workers in turn
        while any(k < len(b) for b in per_worker):
            batches += [b[k] for b in per_worker if k < len(b)]
            k += 1
        self._cache = {epoch: batches}
        return batches


# ----------------------------------------------------------------- viewer --

def to_jpeg_b64(img: Image.Image | torch.Tensor, quality: int = 88) -> str:
    if isinstance(img, torch.Tensor):
        arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def clean(v):
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 3)
    return v


class Viewer:
    def __init__(self, cfg: TrainConfig, vocab: dict, checkpoint: Path | None):
        self.cfg = cfg
        self.stream = TrainingStream(cfg)
        self.dishes, self.ingredients = vocab["dishes"], vocab["ingredients"]
        # transforms without normalisation, so the tensors can be shown as images; both crop modes for comparing
        self.tfs = {(crop, train): build_transform(dataclasses.replace(cfg, crop=crop), train, mean=[0, 0, 0], std=[1, 1, 1])
                    for crop in ("center", "pad") for train in (True, False)}
        self.eval_tf = self.tfs[(cfg.crop, False)]
        self.splits: dict[str, object] = {}
        self.model = self.norm = self.log_std_scale = None
        if checkpoint:
            from .models import FoodModel
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            c = ck["config"]
            self.model = FoodModel(c["backbone"], len(ck["dishes"]), len(ck["ingredients"]), pretrained=False,
                                   hidden=c["hidden"], dropout=c["dropout"], mode=c["head_mode"], rec_hidden=c.get("rec_hidden", 0)).eval()
            self.model.load_state_dict(ck["model"])
            self.norm = ck["normalization"]
            self.log_std_scale = ck.get("log_std_scale", 1.0)
            self.pred_dishes, self.pred_ingredients = ck["dishes"], ck["ingredients"]

    def split(self, key: str):
        if key not in self.splits:
            name, split = key.split(":")
            ds = build_dataset(name, split, self.cfg, False)
            if ds is not None:
                ds.transform = None
            self.splits[key] = ds
        return self.splits[key]

    def info(self) -> dict:
        sources = ["train"] + [f"{n}:{s}" for n in self.cfg.datasets for s in ("val", "test")]
        return {"config": {k: getattr(self.cfg, k) for k in ("backbone", "datasets", "sampling", "augment", "crop",
                                                             "image_size", "batch_size", "workers", "seed", "epochs")},
                "sources": sources, "vocab": {"dishes": len(self.dishes), "ingredients": len(self.ingredients)},
                "train_records": {n: len(ds.records) for n, ds in self.stream.sets.items()},
                "predict": self.model is not None}

    def source_size(self, source: str, epoch: int) -> int:
        if source == "train":
            return sum(len(b) for b in self.stream.epoch(epoch))
        ds = self.split(source)
        return len(ds.records) if ds is not None else 0

    def crop_box(self, size: tuple[int, int], crop: str) -> list[float] | None:
        """Part of the stored photo the model sees, as [x, y, w, h] fractions (None: whole image)."""
        if crop == "pad":
            return None
        w, h = size
        s = min(w, h)
        # (with augment "strong" the training crop is random; the box shows the centre position)
        return [(w - s) / 2 / w, (h - s) / 2 / h, s / w, s / h]

    def sample(self, source: str, index: int, epoch: int, views: int, crop: str | None = None) -> dict:
        crop = crop if crop in ("center", "pad") else self.cfg.crop
        out: dict = {"source": source, "index": index, "crop": crop, "crop_trained": self.cfg.crop}
        if source == "train":
            batches = self.stream.epoch(epoch)
            total = sum(len(b) for b in batches)
            index %= max(total, 1)
            b, pos = divmod(index, self.cfg.batch_size)
            name, rec = batches[b][pos]
            ds = self.stream.sets[name]
            mix = Counter(n for n, _ in batches[b])
            epoch_mix = Counter(n for batch in batches for n, _ in batch)
            out |= {"index": index, "total": total, "epoch": epoch, "batch": b, "n_batches": len(batches),
                    "pos": pos, "batch_mix": dict(mix), "epoch_mix": dict(epoch_mix)}
            tf, draws = self.tfs[(crop, True)], views
        else:
            ds = self.split(source)
            index %= max(len(ds.records), 1)
            rec, name = ds.records[index], ds.name
            out |= {"index": index, "total": len(ds.records)}
            tf, draws = self.tfs[(crop, False)], 1
        sample = ds._load(rec)
        image = sample["image"]
        out["original"] = to_jpeg_b64(image)
        out["original_size"] = list(image.size)
        out["crop_box"] = self.crop_box(image.size, crop)
        out["kept_pct"] = round(100 * min(image.size) / max(image.size)) if crop == "center" else 100
        out["inputs"] = [to_jpeg_b64(tf(image)) for _ in range(draws)]
        out["transform"] = "train (random draws)" if source == "train" else "eval (deterministic)"
        out |= self.describe(name, sample)
        if self.model is not None:
            out["prediction"] = self.predict(image)
        return out

    def describe(self, name: str, s: dict) -> dict:
        t = {k: clean(float(s[k])) if s.get(k) is not None else None for k in TARGET_KEYS}
        mass = t["mass_g"]
        per100 = {}
        if mass and mass > 0:
            for k in ("protein_g", "fat_g", "carb_g"):
                if t[k] is not None:
                    per100[k] = round(min(100.0, max(0.0, t[k] / mass * 100)), 2)  # Huber target, clamped like the loss
        labels = None
        found = components(s)
        if found is not None:
            labels = {}
            for key, names, vocab in (("dishes", found[0], self.dishes), ("ingredients", found[1], self.ingredients)):
                known = set(vocab)
                labels[key] = [{"name": n, "target": n if n in known else "other"} for n in names]
        meta = {k: s.get(k) for k in ("dish_id", "image_id", "dish_name", "camera", "frame") if s.get(k) is not None}
        if s.get("ingredients") and s.get("ingredient_grams"):
            meta["ingredient_grams"] = {i: round(g, 1) for i, g in zip(s["ingredients"], s["ingredient_grams"])}
        if s.get("food_classes") and s.get("food_class_grams"):
            meta["food_class_grams"] = {i: round(g, 1) for i, g in zip(s["food_classes"], s["food_class_grams"])}
        return {"dataset": name, "targets": t, "log_mass": round(math.log(mass), 4) if mass and mass > 0 else None,
                "per_100g": per100, "labels": labels, "has_labels": labels is not None,
                "primary": primary_label(s), "meta": meta}

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        x = self.eval_tf(image)
        x = (x - torch.tensor(self.norm["mean"]).view(3, 1, 1)) / torch.tensor(self.norm["std"]).view(3, 1, 1)
        out = self.model(x.unsqueeze(0))
        tv = out["targets"][0].tolist()
        log_std = float((0.5 * out["log_mass_var"][0].clamp(-7, 7)).exp()) * self.log_std_scale
        lm = float(out["log_mass"][0])

        def top(logits, names):
            p = torch.sigmoid(logits[0])
            idx = torch.argsort(p, descending=True)[:5]
            return [{"name": names[i], "p": round(float(p[i]), 3)} for i in idx if p[i] >= 0.1]
        return {"targets": {k: round(v, 2) for k, v in zip(TARGET_KEYS, tv)},
                "mass_range90": [round(math.exp(lm - 1.645 * log_std), 1), round(math.exp(lm + 1.645 * log_std), 1)],
                "dishes": top(out["dish_logits"], self.pred_dishes),
                "ingredients": top(out["ingredient_logits"], self.pred_ingredients)}


# -------------------------------------------------------------------- web --

PAGE = (Path(__file__).with_name("visualize.html")).read_text


def serve(viewer: Viewer, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    return self.send(200, PAGE().encode(), "text/html; charset=utf-8")
                if url.path == "/api/info":
                    body = viewer.info()
                elif url.path == "/api/sample":
                    source = q.get("source", "train")
                    epoch = max(1, int(q.get("epoch", 1)))
                    index = int(q.get("i", 0))
                    if q.get("random"):
                        index = random.randrange(max(1, viewer.source_size(source, epoch)))
                    body = viewer.sample(source, index, epoch, min(8, max(1, int(q.get("views", 4)))), q.get("crop"))
                else:
                    return self.send(404, b"not found", "text/plain")
                self.send(200, json.dumps(body).encode(), "application/json")
            except Exception as e:  # noqa: BLE001 - show the error in the page
                self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"viewer on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, help="run directory: take its config and label vocabulary (results.json, labels.json)")
    p.add_argument("--predict", action="store_true", help="also show the prediction of <run>/best.pt (CPU)")
    p.add_argument("--checkpoint", type=Path, help="checkpoint for --predict (default <run>/best.pt)")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--sampling", nargs="*")
    p.add_argument("--augment", choices=["none", "basic", "strong"])
    p.add_argument("--crop", choices=["center", "pad"])
    p.add_argument("--image-size", type=int)
    p.add_argument("--workers", type=int, help="DataLoader workers of the run (changes the order)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()

    known = {f.name for f in dataclasses.fields(TrainConfig)}
    base = {"local_images": True}
    if a.run:
        base |= {k: v for k, v in json.loads((a.run / "results.json").read_text())["config"].items() if k in known}
    for key in ("datasets", "sampling", "augment", "crop", "image_size", "workers"):
        if getattr(a, key) is not None:
            base[key] = getattr(a, key)
    cfg = TrainConfig(**base)

    if a.run and (a.run / "labels.json").exists():
        vocab = json.loads((a.run / "labels.json").read_text())
    else:  # same rule as trainer.train
        counts = [Counter(), Counter()]
        for name in cfg.datasets:
            ds = build_dataset(name, "train", cfg, False)
            for rec in {r["dish_id"]: r for r in ds.records}.values():
                for counter, names in zip(counts, components(rec) or ([], [])):
                    counter.update(names)
        vocab = {k: [n for n, c in counter.most_common() if c >= cfg.min_label_count] + ["other"]
                 for k, counter in zip(("dishes", "ingredients"), counts)}

    checkpoint = a.checkpoint or (a.run / "best.pt" if a.run and a.predict else None)
    if a.predict and (checkpoint is None or not checkpoint.exists()):
        p.error(f"--predict needs a checkpoint ({checkpoint} not found)")
    serve(Viewer(cfg, vocab, checkpoint if a.predict or a.checkpoint else None), a.host, a.port)


if __name__ == "__main__":
    main()
