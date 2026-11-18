"""Train and evaluate a ``FoodModel`` (Stage 1, RGB) on the streaming datasets.

Command line (run from the repository root)::

    uv run python -m training.trainer --run-name mnv3_baseline --backbone mobilenetv3_large
    uv run python -m training.trainer --help                      # all options (= fields of TrainConfig)
    uv run python -m training.trainer --run-name smoke --limit 64 --epochs 2 --freeze-epochs 1   # quick check

Python::

    from training.trainer import TrainConfig, train
    results = train(TrainConfig(run_name="mnv3_baseline", backbone="mobilenetv3_large"))

What a run does
    1. Builds train / val / test splits (always by dish) of the chosen datasets.
       Images are streamed on first use and kept in ``image_cache_dir``, so only
       the first epoch needs the network.
    2. Trains with the backbone frozen for ``freeze_epochs`` epochs, then end to end.
       After every epoch the model is evaluated on the val split; the epoch with the
       lowest val score (mean relative error over mass, kcal, fat, carbs, protein)
       is kept as ``best.pt``.
    3. Evaluates ``best.pt`` on the test split and compares it with two trivial
       baselines (training-set mean, median of the training dishes with the same
       main component).

Labels of the two recognition outputs
    ``datasets/label_map.csv`` maps the labels of all datasets to a common vocabulary
    of *dishes* (processed / prepared items) and *ingredients* (plain foods). Both
    outputs are trained on every dataset that has component labels. Labels with
    fewer than ``min_label_count`` training dishes are collected in "other".

Files written to ``runs/<run_name>/`` (everything needed for a report)
    results.json          config, data sizes, training summary, val and test metrics, baselines
    summary.md            the main test results as Markdown tables
    history.jsonl         one JSON line per epoch: losses, learning rate, val metrics, seconds
    predictions_test.csv  one row per test photo: labels, predictions, 90 % mass interval
    labels.json           dish and ingredient vocabularies of the two recognition outputs
    best.pt, last.pt      checkpoints (weights, config, label vocabularies)

Metrics
    ``mae``           mean absolute error in grams / kcal
    ``rel_err_pct``   MAE as percentage of the mean true value (as in the Nutrition5k paper)
    ``photo``         every photo is one prediction
    ``dish_median``   median of the predictions of all photos of a dish
    ``calibration``   share of true masses inside the predicted 68 % / 90 % range
    ``f1_micro``, ``top1_acc``  multi-label recognition (threshold 0.5; top-1 = the highest
                      score is a true label, over the photos that have a label of that kind)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from .datasets import (CACHE_DIR, TARGET_KEYS, Interleave, LocalFoodDataset, MMFood100K, Nutrition5k,
                       Nutrition5kSideAngles, NutritionVerseReal, local_available)
from .datasets.labels import components
from .label_quality import clean as clean_labels
from .models import BACKBONES, FoodModel

DATASETS = ("nutrition5k", "nutritionverse_real", "nutrition5k_side", "mm_food_100k")
Z90 = 1.645  # half width of a 90 % interval in standard deviations


@dataclass
class TrainConfig:
    run_name: str = "run"
    out_dir: str = "runs"
    backbone: str = "mobilenetv3_large"            # one of models.BACKBONES
    datasets: list[str] = field(default_factory=lambda: ["nutrition5k", "nutritionverse_real"])
    # val score + test; side views are 60 % of the score because real photos are rarely taken straight from above
    eval_datasets: list[str] = field(default_factory=lambda: ["nutrition5k", "nutritionverse_real", "nutrition5k_side"])
    eval_weights: list[str] = field(default_factory=lambda: ["nutrition5k=0.2", "nutritionverse_real=0.2",
                                                             "nutrition5k_side=0.6"])  # "<dataset>=<w>", missing = 1
    sampling: list[str] = field(default_factory=list)   # "<dataset>=<factor>": records drawn per epoch (2 = twice, 0.3 = 30 %)
    augment: str = "basic"                         # none | basic | strong (never changes the scale)
    crop: str = "center"                           # center (crop to square) | pad (pad to square, whole image kept)
    head_mode: str = "factorized"                  # factorized (amount x composition) | direct (comparison)
    hidden: int = 512
    dropout: float = 0.2
    ema: float = 0.0                               # > 0: evaluate / keep an exponential moving average of the weights
    tta: bool = False                              # evaluation: average with the horizontally flipped image
    init_from: str = ""                            # checkpoint to start from (layers with matching shapes)
    freeze_bn: bool = False                        # backbone BatchNorm always uses its ImageNet statistics
    unfreeze_warmup: float = 0.0                   # epochs of linear backbone-lr warm-up after unfreezing
    clean_labels: bool = True                      # drop records with impossible labels (label_quality.py)
    local_images: bool = False                     # read Nutrition5k / NutritionVerse-Real from training.prefetch convert
    keep_last: bool = False                        # keep last.pt after training (disk space: only best.pt by default)
    image_size: int = 224
    epochs: int = 30
    freeze_epochs: int = 3                         # epochs with frozen backbone at the start
    batch_size: int = 64
    lr: float = 1e-3                               # head; decays with a cosine schedule
    backbone_lr_mult: float = 0.1                  # backbone learning rate = lr * this
    weight_decay: float = 0.01
    composition_weight: float = 0.2                # loss weights (the mass loss has weight 1)
    dish_weight: float = 1.0
    ingredient_weight: float = 1.0
    min_label_count: int = 5                       # labels with fewer training dishes become "other"
    frames_per_video: int = 4                      # only for nutrition5k_side
    workers: int = 8
    amp: bool = True                               # mixed precision on the GPU
    pretrained: bool = True
    seed: int = 0
    limit: int = 0                                 # > 0: use only this many records per dataset and split
    image_cache_dir: str = str(CACHE_DIR / "images")   # "" = always stream
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------- data --

def primary_label(sample: dict) -> str:
    """Main component of a dish: the heaviest food item / ingredient, or the dish name."""
    for names, grams in (("food_classes", "food_class_grams"), ("ingredients", "ingredient_grams")):
        if sample.get(names) and sample.get(grams):
            return sample[names][int(np.argmax(sample[grams]))]
    return sample.get("dish_name", "")


class PadSquare:
    """Pad the shorter side so the image becomes square (the whole image is kept)."""

    def __init__(self, fill=(124, 116, 104)):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        out = Image.new("RGB", (side, side), self.fill)
        out.paste(img, ((side - w) // 2, (side - h) // 2))
        return out


def build_transform(cfg: TrainConfig, train: bool, mean, std):
    """Resize + crop / pad to a square: the scale of the food is never changed by augmentation."""
    size = cfg.image_size
    if cfg.crop == "pad":
        steps = [PadSquare(), transforms.Resize((size, size))]
    else:
        steps = [transforms.Resize(size), transforms.RandomCrop(size) if train and cfg.augment == "strong"
                 else transforms.CenterCrop(size)]
    if train and cfg.augment == "basic":
        steps += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.02)]
    elif train and cfg.augment == "strong":
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15, fill=(124, 116, 104)),
                  transforms.ColorJitter(0.3, 0.3, 0.3, 0.03), transforms.RandomGrayscale(0.05),
                  transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.2)]
    elif train and cfg.augment != "none":
        raise ValueError(f"unknown augment {cfg.augment!r}")
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def build_dataset(name: str, split: str, cfg: TrainConfig, train: bool):
    common = dict(shuffle=train, seed=cfg.seed, draft_size=cfg.image_size,
                  image_cache_dir=cfg.image_cache_dir or None)
    if cfg.local_images and name in ("nutrition5k", "nutritionverse_real") and local_available(name, split):
        ds = LocalFoodDataset(name, split, shuffle=train, seed=cfg.seed)
    elif name == "nutrition5k":
        ds = Nutrition5k(split, **common)
    elif name == "nutrition5k_side":
        if local_available(name, split):  # prefetched fixed frames (training.prefetch)
            ds = LocalFoodDataset(name, split, shuffle=train, seed=cfg.seed)
        else:
            ds = Nutrition5kSideAngles(split, frames_per_video=cfg.frames_per_video, **common)
    elif name == "nutritionverse_real":
        ds = NutritionVerseReal(split, split_source="hash", **common)
    elif name == "mm_food_100k":
        if split == "test":
            return None  # weak labels: used for training and validation only
        ds = (LocalFoodDataset(name, split, shuffle=train, seed=cfg.seed) if local_available(name, split)
              else MMFood100K(split, **common))
    else:
        raise ValueError(f"unknown dataset {name!r}, choose from {DATASETS}")
    if cfg.clean_labels and ds.records:
        ds.records, dropped = clean_labels(ds.records)
        if dropped:
            dishes = sorted({r["dish_id"] for r, _ in dropped})
            print(f"{name}/{split}: dropped {len(dropped)} records of {len(dishes)} dishes with impossible labels "
                  f"({'; '.join(sorted({why for _, why in dropped})[:4])})", flush=True)
    if cfg.limit:
        records = list(ds.records)
        random.Random(cfg.seed).shuffle(records)
        ds.records = records[:cfg.limit]
    return ds


def parse_sampling(cfg: TrainConfig) -> dict[str, float]:
    out = {}
    for item in cfg.sampling:
        name, _, factor = item.partition("=")
        if name not in DATASETS or not factor:
            raise ValueError(f"bad sampling entry {item!r}, expected <dataset>=<factor>")
        out[name] = float(factor)
    return out


def resample(ds, factor: float, rng: random.Random) -> None:
    """Draw ``factor`` x the records of ``ds`` for this epoch (whole copies + a random part)."""
    base = ds.all_records = getattr(ds, "all_records", None) or list(ds.records)
    whole, part = int(factor), factor - int(factor)
    ds.records = base * whole + rng.sample(base, round(part * len(base)))


class Collate:
    """Batch of samples -> tensors, with multi-hot labels for the two recognition heads."""

    def __init__(self, dishes: list[str], ingredients: list[str]):
        self.index = [{n: i for i, n in enumerate(dishes)}, {n: i for i, n in enumerate(ingredients)}]

    def __call__(self, batch: list[dict]) -> dict:
        n = len(batch)
        hot = [torch.zeros(n, len(index)) for index in self.index]
        has_labels = torch.zeros(n, dtype=torch.bool)
        for i, s in enumerate(batch):
            if (found := components(s)) is None:
                continue
            has_labels[i] = True
            for names, index, target in zip(found, self.index, hot):
                for name in names:
                    target[i, index.get(name, index["other"])] = 1
        return {
            "image": torch.stack([s["image"] for s in batch]),
            "targets": torch.tensor([[s[k] for k in TARGET_KEYS] for s in batch], dtype=torch.float32),
            "dishes": hot[0], "ingredients": hot[1], "has_labels": has_labels,
            "dataset": [s["dataset"] for s in batch],
            "dish_id": [s["dish_id"] for s in batch],
            "image_id": [s["image_id"] for s in batch],
            "primary": [primary_label(s) for s in batch],
        }


def loader(ds, cfg: TrainConfig, collate: Collate, train: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.workers, collate_fn=collate, drop_last=train)


# ----------------------------------------------------------------------- loss --

def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)


def compute_loss(out: dict, batch: dict, cfg: TrainConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """Gaussian NLL on log(mass) + Huber on the composition per 100 g + BCE on the label heads.

    Every part only uses the samples that carry the corresponding label.
    """
    t = batch["targets"]
    mass = t[:, 0]
    has_mass = torch.isfinite(mass) & (mass > 0)
    mu, log_var = out["log_mass"].float(), out["log_mass_var"].float().clamp(-7, 7)
    nll = 0.5 * (torch.exp(-log_var) * (mass.clamp_min(1e-3).log() - mu) ** 2 + log_var)
    parts = {"mass_nll": _masked_mean(nll, has_mass)}

    if "log_totals" in out:  # direct regression: Huber on log(1 + total) of kcal, fat, carbs, protein
        totals = t[:, 1:]
        valid = torch.isfinite(totals) & (totals >= 0)
        huber = F.smooth_l1_loss(out["log_totals"].float(), totals.nan_to_num().clamp_min(0).log1p(), reduction="none")
        parts["composition"] = _masked_mean(huber, valid)
        comp_weight = 1.0
    else:
        per_100g = t[:, [4, 2, 3]] / mass.unsqueeze(1) * 100  # protein, fat, carbs like the head
        valid = has_mass.unsqueeze(1) & torch.isfinite(per_100g)
        huber = F.smooth_l1_loss(out["per_100g"].float(), per_100g.nan_to_num().clamp(0, 100), reduction="none")
        parts["composition"] = _masked_mean(huber, valid)
        comp_weight = cfg.composition_weight

    for key, logits, weight_key in (("dishes", "dish_logits", "dish"), ("ingredients", "ingredient_logits", "ingredient")):
        if logits in out:
            bce = F.binary_cross_entropy_with_logits(out[logits].float(), batch[key], reduction="none").mean(1)
            parts[weight_key] = _masked_mean(bce, batch["has_labels"])

    total = (parts["mass_nll"] + comp_weight * parts["composition"]
             + cfg.dish_weight * parts.get("dish", 0) + cfg.ingredient_weight * parts.get("ingredient", 0))
    return total, {k: v.item() for k, v in parts.items()} | {"total": total.item()}


# ----------------------------------------------------------------- evaluation --

@torch.no_grad()
def predict(model: FoodModel, data: DataLoader, cfg: TrainConfig) -> dict:
    """Run the model over a loader and collect predictions and labels as numpy arrays."""
    model.eval()
    cols = defaultdict(list)
    losses, n = defaultdict(float), 0
    for batch in data:
        dev = {k: v.to(cfg.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.autocast(cfg.device, enabled=cfg.amp and cfg.device == "cuda"):
            out = model(dev["image"])
            if cfg.tta:
                flipped = model(dev["image"].flip(3))
                out = {k: (v + flipped[k]) / 2 for k, v in out.items()}
        _, parts = compute_loss(out, dev, cfg)
        b = len(batch["image_id"])
        n += b
        for k, v in parts.items():
            losses[k] += v * b
        cols["pred"].append(out["targets"].float().cpu().numpy())
        cols["true"].append(batch["targets"].numpy())
        cols["log_mass"].append(out["log_mass"].float().cpu().numpy())
        cols["log_mass_std"].append((0.5 * out["log_mass_var"].float().clamp(-7, 7)).exp().cpu().numpy())
        cols["has_labels"].append(batch["has_labels"].numpy())
        for key, logits in (("dishes", "dish_logits"), ("ingredients", "ingredient_logits")):
            if logits in out:
                cols[f"{key}_logits"].append(out[logits].float().cpu().numpy())
                cols[f"{key}_true"].append(batch[key].numpy())
        for k in ("dataset", "dish_id", "image_id", "primary"):
            cols[k] += batch[k]
    res = {k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)) for k, v in cols.items()}
    res["loss"] = {k: v / max(n, 1) for k, v in losses.items()}
    return res


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    out = {}
    for i, key in enumerate(TARGET_KEYS):
        m = np.isfinite(true[:, i])
        if not m.any():
            continue
        mae = float(np.abs(pred[m, i] - true[m, i]).mean())
        out[key] = {"mae": round(mae, 2), "rel_err_pct": round(100 * mae / float(true[m, i].mean()), 2)}
    out["mean_rel_err_pct"] = round(float(np.mean([v["rel_err_pct"] for v in out.values()])), 2) if out else math.nan
    out["n"] = int(len(true))
    return out


def dish_median(pred: np.ndarray, true: np.ndarray, dish_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine the photos of a dish by the median of their predictions."""
    groups = defaultdict(list)
    for i, d in enumerate(dish_ids):
        groups[d].append(i)
    idx = list(groups.values())
    return np.stack([np.median(pred[i], axis=0) for i in idx]), np.stack([true[i[0]] for i in idx])


def label_metrics(logits: np.ndarray, true: np.ndarray) -> dict:
    pred = logits > 0
    tp = float((pred & (true > 0)).sum())
    precision, recall = tp / max(float(pred.sum()), 1.0), tp / max(float(true.sum()), 1.0)
    labelled = true.sum(1) > 0  # photos that have at least one label of this kind
    top1 = true[labelled][np.arange(int(labelled.sum())), logits[labelled].argmax(1)] > 0
    return {"f1_micro": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "top1_acc": round(float(top1.mean()), 4) if labelled.any() else None,
            "n": int(len(true)), "n_with_label": int(labelled.sum())}


def std_scale(log_mass: np.ndarray, log_std: np.ndarray, true_mass: np.ndarray) -> float:
    """Factor for the predicted log-std so that 90 % of the true masses fall inside the 90 % range."""
    m = np.isfinite(true_mass) & (true_mass > 0)
    z = np.abs(np.log(true_mass[m]) - log_mass[m]) / log_std[m]
    return float(np.quantile(z, 0.9) / Z90)


def calibration(log_mass: np.ndarray, log_std: np.ndarray, true_mass: np.ndarray) -> dict:
    m = np.isfinite(true_mass) & (true_mass > 0)
    z = (np.log(true_mass[m]) - log_mass[m]) / log_std[m]
    return {"coverage_68": round(float((np.abs(z) < 1).mean()), 4), "coverage_90": round(float((np.abs(z) < Z90).mean()), 4),
            "z_std": round(float(z.std()), 3),
            "median_90_range_factor": round(float(np.exp(Z90 * np.median(log_std[m]))), 3)}


class Baselines:
    """Trivial predictors fitted on the training records of one dataset."""

    def __init__(self, records: list[dict]):
        t = np.array([[r[k] for k in TARGET_KEYS] for r in records], dtype=np.float64)
        self.mean = np.nanmean(t, axis=0)
        self.median = np.nanmedian(t, axis=0)
        by_label = defaultdict(list)
        for r, row in zip(records, t):
            by_label[primary_label(r)].append(row)
        self.label_median = {k: np.nanmedian(np.array(v), axis=0) for k, v in by_label.items()}

    def predict_mean(self, n: int) -> np.ndarray:
        return np.tile(self.mean, (n, 1))

    def predict_class_median(self, primary: np.ndarray) -> np.ndarray:
        rows = np.stack([self.label_median.get(p, self.median) for p in primary])
        return np.where(np.isfinite(rows), rows, self.median)


def evaluate(res: dict, baselines: dict[str, Baselines], detailed: bool) -> dict:
    """Metrics per dataset from the output of ``predict``."""
    report = {}
    for name in sorted(set(res["dataset"])):
        m = res["dataset"] == name
        pred, true = res["pred"][m], res["true"][m]
        r = {"photo": regression_metrics(pred, true),
             "dish_median": regression_metrics(*dish_median(pred, true, res["dish_id"][m])),
             "n_photos": int(m.sum()), "n_dishes": len(set(res["dish_id"][m]))}
        if name in baselines:
            r["baseline_mean"] = regression_metrics(baselines[name].predict_mean(len(true)), true)
            r["baseline_class_median"] = regression_metrics(baselines[name].predict_class_median(res["primary"][m]), true)
            r["beats_baselines"] = {k: r["photo"][k]["mae"] < min(r["baseline_mean"][k]["mae"], r["baseline_class_median"][k]["mae"])
                                    for k in TARGET_KEYS if k in r["photo"]}
        r["calibration"] = calibration(res["log_mass"][m], res["log_mass_std"][m], true[:, 0])
        for key in ("dishes", "ingredients"):
            if f"{key}_logits" in res and (has := m & res["has_labels"]).any():
                r[key] = label_metrics(res[f"{key}_logits"][has], res[f"{key}_true"][has])
        if detailed:  # error per main component, most frequent first (for the error analysis)
            rows = []
            for label, count in Counter(res["primary"][m]).most_common(30):
                if count >= 5:
                    lm = res["primary"][m] == label
                    rows.append({"label": label, "n": count} | {k: v for k, v in regression_metrics(pred[lm], true[lm]).items()
                                                                if k in ("mass_g", "kcal")})
            r["by_primary_label"] = rows
        report[name] = r
    return report


def parse_weights(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, _, w = item.partition("=")
        if name not in DATASETS or not w:
            raise ValueError(f"bad eval weight {item!r}, expected <dataset>=<weight>")
        out[name] = float(w)
    return out


def val_score(report: dict, weights: dict[str, float] | None = None) -> float:
    """Model selection: mean relative error of the five targets, weighted mean over the datasets."""
    weights = weights or {}
    return float(np.average([r["photo"]["mean_rel_err_pct"] for r in report.values()],
                            weights=[weights.get(n, 1.0) for n in report]))


# -------------------------------------------------------------------- outputs --

def write_predictions(path: Path, res: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "dish_id", "image_id", "primary_label"] + [f"true_{k}" for k in TARGET_KEYS]
                   + [f"pred_{k}" for k in TARGET_KEYS] + ["pred_mass_lo90_g", "pred_mass_hi90_g"])
        for i in range(len(res["image_id"])):
            lo, hi = (math.exp(res["log_mass"][i] + s * Z90 * res["log_mass_std"][i]) for s in (-1, 1))
            w.writerow([res["dataset"][i], res["dish_id"][i], res["image_id"][i], res["primary"][i]]
                       + [f"{v:.2f}" for v in res["true"][i]] + [f"{v:.2f}" for v in res["pred"][i]] + [f"{lo:.1f}", f"{hi:.1f}"])


def write_summary(path: Path, results: dict) -> None:
    cfg, tr = results["config"], results["training"]
    lines = [f"# Run `{cfg['run_name']}`", "",
             f"- Model: {cfg['backbone']} ({results['model']['parameters_m']} M parameters), {cfg['image_size']} px",
             f"- Training data: {', '.join(cfg['datasets'])}",
             f"- Epochs: {tr['epochs']} ({cfg['freeze_epochs']} with frozen backbone), best epoch {tr['best_epoch']} "
             f"(val score {tr['best_val_score']}), {tr['minutes']} min on {tr['device']}", ""]
    for name, r in results["test"].items():
        lines += [f"## Test: {name} ({r['n_photos']} photos, {r['n_dishes']} dishes)", "",
                  "MAE (relative error in % of the mean true value)", "",
                  "| | " + " | ".join(TARGET_KEYS) + " | mean rel. % |", "|---|" + "---|" * (len(TARGET_KEYS) + 1)]
        for title, key in (("Model, per photo", "photo"), ("Model, median per dish", "dish_median"),
                           ("Baseline: training mean", "baseline_mean"), ("Baseline: median of same main component", "baseline_class_median")):
            if key in r:
                cells = [f"{r[key][k]['mae']} ({r[key][k]['rel_err_pct']} %)" if k in r[key] else "-" for k in TARGET_KEYS]
                lines.append(f"| {title} | " + " | ".join(cells) + f" | {r[key]['mean_rel_err_pct']} |")
        c = r["calibration"]
        lines += ["", f"- Mass uncertainty: {100 * c['coverage_68']:.0f} % of the true masses inside the 68 % range, "
                      f"{100 * c['coverage_90']:.0f} % inside the 90 % range (typical 90 % range: ÷/× {c['median_90_range_factor']})"]
        for key, title in (("dishes", "Dishes"), ("ingredients", "Ingredients")):
            if key in r:
                lines.append(f"- {title}: micro-F1 {r[key]['f1_micro']}, top-1 accuracy {r[key]['top1_acc']} "
                             f"({r[key]['n_with_label']} of {r[key]['n']} photos have such a label)")
        if "beats_baselines" in r:
            lost = [k for k, v in r["beats_baselines"].items() if not v]
            lines.append("- Beats both baselines on every target" if not lost else f"- Does NOT beat the baselines on: {', '.join(lost)}")
        lines.append("")
    path.write_text("\n".join(lines))


# ------------------------------------------------------------------- training --

def train(cfg: TrainConfig) -> dict:
    """Run training and evaluation as described in the module docstring; returns the content of results.json."""
    for name in cfg.datasets + cfg.eval_datasets:
        if name not in DATASETS:
            raise ValueError(f"unknown dataset {name!r}, choose from {DATASETS}")
    sampling = parse_sampling(cfg)
    weights = parse_weights(cfg.eval_weights)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # data (records only; images are fetched while iterating)
    train_sets = {n: build_dataset(n, "train", cfg, True) for n in cfg.datasets}
    eval_sets = {split: {n: ds for n in cfg.eval_datasets if (ds := build_dataset(n, split, cfg, False)) is not None}
                 for split in ("val", "test")}

    # label vocabularies: dishes / ingredients that occur in enough training dishes, plus "other"
    counts = [Counter(), Counter()]
    for ds in train_sets.values():
        for rec in {r["dish_id"]: r for r in ds.records}.values():
            for counter, names in zip(counts, components(rec) or ([], [])):
                counter.update(names)
    dishes, ingredients = ([n for n, c in counter.most_common() if c >= cfg.min_label_count] + ["other"] for counter in counts)
    (run_dir / "labels.json").write_text(json.dumps({"dishes": dishes, "ingredients": ingredients}, indent=1))

    model = FoodModel(cfg.backbone, len(dishes), len(ingredients), pretrained=cfg.pretrained,
                      hidden=cfg.hidden, dropout=cfg.dropout, mode=cfg.head_mode).to(cfg.device)
    if cfg.init_from:  # e.g. pre-trained on MM-Food-100K: take every layer whose shape still fits
        state = torch.load(cfg.init_from, map_location=cfg.device)["model"]
        own = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(loaded, strict=False)
        print(f"[{cfg.run_name}] init from {cfg.init_from}: {len(loaded)}/{len(own)} tensors", flush=True)
    pre = model.backbone.pretrained_cfg
    norm = {"mean": list(pre["mean"]), "std": list(pre["std"])}
    collate = Collate(dishes, ingredients)
    for ds in train_sets.values():
        ds.transform = build_transform(cfg, True, **norm)
    for split in eval_sets.values():
        for ds in split.values():
            ds.transform = build_transform(cfg, False, **norm)
    baselines = {n: Baselines(ds.records) for n, ds in train_sets.items()}
    for n in cfg.eval_datasets:  # eval-only datasets: baselines from their own training split
        if n not in baselines and (ds := build_dataset(n, "train", cfg, False)) is not None:
            baselines[n] = Baselines(ds.records)
    mixed = Interleave(list(train_sets.values()), cfg.seed)
    data = {"train": {n: len(ds) for n, ds in train_sets.items()}} | {s: {n: len(ds) for n, ds in d.items()} for s, d in eval_sets.items()}
    print(f"[{cfg.run_name}] {cfg.backbone} on {cfg.device} | records {data} | sampling {sampling}", flush=True)

    ema = None
    if cfg.ema > 0:
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.ema), use_buffers=True)
    eval_model = ema.module if ema is not None else model

    def run_eval(split: str, detailed: bool = False):
        parts = [predict(eval_model, loader(ds, cfg, collate, False), cfg) for ds in eval_sets[split].values() if len(ds)]
        res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "loss"}
        loss = float(np.average([p["loss"]["total"] for p in parts], weights=[len(p["image_id"]) for p in parts]))
        return res, evaluate(res, baselines, detailed), loss

    # optimisation
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": head_params, "lr": cfg.lr},
                                   {"params": model.backbone.parameters(), "lr": cfg.lr * cfg.backbone_lr_mult}],
                                  weight_decay=cfg.weight_decay)
    base_lrs = [cfg.lr, cfg.lr * cfg.backbone_lr_mult]
    steps_per_epoch = max(1, len(mixed) // cfg.batch_size)  # estimate for the per-step schedule

    def set_lr(epoch: int, step: int) -> None:
        """Cosine decay per step; the backbone additionally warms up linearly after unfreezing."""
        progress = min(1.0, (epoch - 1 + step / steps_per_epoch) / max(cfg.epochs, 1))
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        warm = 1.0
        if cfg.unfreeze_warmup > 0:
            since = (epoch - 1 - cfg.freeze_epochs) * steps_per_epoch + step
            warm = min(1.0, max(0.0, since) / (cfg.unfreeze_warmup * steps_per_epoch))
        optimizer.param_groups[0]["lr"] = base_lrs[0] * cos
        optimizer.param_groups[1]["lr"] = base_lrs[1] * cos * warm
    use_amp = cfg.amp and cfg.device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    def checkpoint(path: Path, epoch: int, **extra) -> None:
        torch.save({"model": eval_model.state_dict(), "config": dataclasses.asdict(cfg), "epoch": epoch,
                    "dishes": dishes, "ingredients": ingredients, "normalization": norm} | extra, path)

    best = {"epoch": 0, "score": math.inf}
    history = (run_dir / "history.jsonl").open("w")
    rng = random.Random(cfg.seed)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        for name, factor in sampling.items():
            if name in train_sets:
                resample(train_sets[name], factor, rng)
        model.freeze_backbone(epoch <= cfg.freeze_epochs)
        model.train()
        if cfg.freeze_bn:
            for m in model.backbone.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        mixed.set_epoch(epoch)
        sums, steps, images = defaultdict(float), 0, 0
        for batch in loader(mixed, cfg, collate, True):
            dev = {k: v.to(cfg.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(cfg.device, enabled=use_amp):
                out = model(dev["image"])
            loss, parts = compute_loss(out, dev, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {float(loss)} in epoch {epoch}, step {steps}")
            set_lr(epoch, steps)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update_parameters(model)
            steps, images = steps + 1, images + len(batch["image_id"])
            for k, v in parts.items():
                sums[k] += v
        if steps == 0:
            raise RuntimeError("no training batch: fewer training images than batch_size per worker")

        _, report, val_loss = run_eval("val")
        score = val_score(report, weights)
        row = {"epoch": epoch, "backbone_frozen": epoch <= cfg.freeze_epochs, "lr": optimizer.param_groups[0]["lr"],
               "train_images": images, "train_loss": {k: round(v / steps, 4) for k, v in sums.items()},
               "val_loss": round(val_loss, 4), "val_score": round(score, 2),
               "val": {n: {"mean_rel_err_pct": r["photo"]["mean_rel_err_pct"],
                           "mass_rel_err_pct": r["photo"]["mass_g"]["rel_err_pct"],
                           "kcal_rel_err_pct": r["photo"]["kcal"]["rel_err_pct"]} for n, r in report.items()},
               "seconds": round(time.time() - t0, 1)}
        history.write(json.dumps(row) + "\n")
        history.flush()
        checkpoint(run_dir / "last.pt", epoch)
        if score < best["score"]:
            best = {"epoch": epoch, "score": score}
            checkpoint(run_dir / "best.pt", epoch)
        print(f"[{cfg.run_name}] epoch {epoch:3d}/{cfg.epochs} | train loss {row['train_loss']['total']:.3f} | "
              f"val loss {val_loss:.3f} | val score {score:.1f} % {'*' if best['epoch'] == epoch else ' '} | {row['seconds']:.0f} s", flush=True)
    history.close()

    # final evaluation with the best weights; the uncertainty range is calibrated on val
    eval_model.load_state_dict(torch.load(run_dir / "best.pt", map_location=cfg.device)["model"])
    val_res, val_report, _ = run_eval("val")
    scale = std_scale(val_res["log_mass"], val_res["log_mass_std"], val_res["true"][:, 0])
    test_res, test_report, _ = run_eval("test", detailed=True)
    test_res["log_mass_std"] = test_res["log_mass_std"] * scale
    for name, r in test_report.items():
        m = test_res["dataset"] == name
        r["calibration_raw"] = r["calibration"]
        r["calibration"] = calibration(test_res["log_mass"][m], test_res["log_mass_std"][m], test_res["true"][m, 0])
    checkpoint(run_dir / "best.pt", best["epoch"], log_std_scale=scale)
    write_predictions(run_dir / "predictions_test.csv", test_res)

    results = {
        "config": dataclasses.asdict(cfg),
        "model": {"backbone": cfg.backbone, "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                  "dish_labels": len(dishes), "ingredient_labels": len(ingredients), "normalization": norm,
                  "log_std_scale": round(scale, 4)},
        "data_records": data,
        "training": {"epochs": cfg.epochs, "best_epoch": best["epoch"], "best_val_score": round(best["score"], 2),
                     "minutes": round((time.time() - started) / 60, 1), "device": cfg.device},
        "val": val_report,
        "test": test_report,
        "test_score": round(val_score(test_report, weights), 2),
        "eval_weights": {n: weights.get(n, 1.0) for n in test_report},
        "files": {k: str(run_dir / v) for k, v in {"summary": "summary.md", "history": "history.jsonl",
                  "predictions": "predictions_test.csv", "labels": "labels.json", "best_checkpoint": "best.pt"}.items()},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    if not cfg.keep_last:
        (run_dir / "last.pt").unlink(missing_ok=True)
    write_summary(run_dir / "summary.md", results)
    print(f"[{cfg.run_name}] done in {results['training']['minutes']} min | results: {run_dir / 'results.json'} | "
          f"summary: {run_dir / 'summary.md'}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    choices = {"backbone": sorted(BACKBONES), "augment": ["none", "basic", "strong"], "crop": ["center", "pad"],
               "head_mode": ["factorized", "direct"]}
    for f in dataclasses.fields(TrainConfig):
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif f.name in ("datasets", "eval_datasets"):
            parser.add_argument(flag, nargs="+", default=default, choices=DATASETS)
        elif isinstance(default, list):
            parser.add_argument(flag, nargs="*", default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default, choices=choices.get(f.name))
    train(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
