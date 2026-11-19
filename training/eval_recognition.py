"""Compare dish / ingredient recognition of several runs fairly (no training).

Runs trained with extra recognition data (MM-Food-100K, Food-101) know many more
label names, which makes their plain test metrics incomparable. This script scores
every run

1. on the test sets, restricted to the label names *all* given runs know, and
2. on the Food-101 held-out photos (Western dishes): is the top dish the right one
   (top-1 / top-5), and is the top ingredient one of the dish's typical ingredients.

    uv run python -m training.eval_recognition runs/clean/*_pad_s? runs/mm_diag/rec1024_s? --out report.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .datasets.labels import components
from .models import FoodModel
from .trainer import Collate, TrainConfig, build_dataset, build_transform, label_metrics, loader, predict


def _ap(prob: np.ndarray, true: np.ndarray) -> float:
    """Mean average precision over the labels that occur (ranking quality, threshold-free)."""
    aps = []
    for j in range(true.shape[1]):
        if true[:, j].any():
            t = true[np.argsort(-prob[:, j]), j]
            aps.append(float((np.cumsum(t) / np.arange(1, len(t) + 1))[t].mean()))
    return float(np.mean(aps)) if aps else float("nan")


def load(run: Path):
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    defaults, known = TrainConfig(), {f.name for f in dataclasses.fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in known})
    cfg.eval_datasets, cfg.clean_labels, cfg.workers = defaults.eval_datasets, True, 6
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FoodModel(cfg.backbone, len(ck["dishes"]), len(ck["ingredients"]), pretrained=False, hidden=cfg.hidden,
                      dropout=cfg.dropout, mode=cfg.head_mode, rec_hidden=cfg.rec_hidden).to(cfg.device)
    model.load_state_dict(ck["model"])
    return ck, cfg, model


def evaluate_run(run: Path, shared: dict[str, list[str]]) -> dict:
    ck, cfg, model = load(run)
    collate = Collate(ck["dishes"], ck["ingredients"])
    out = {"n_dishes": len(ck["dishes"]) - 1, "n_ingredients": len(ck["ingredients"]) - 1}
    for name in cfg.eval_datasets:
        ds = build_dataset(name, "test", cfg, False)
        ds.transform = build_transform(cfg, False, **ck["normalization"])
        res = predict(model, loader(ds, cfg, collate, False), cfg)
        h = res["has_labels"].astype(bool)
        for key in ("dishes", "ingredients"):
            idx = [ck[key].index(x) for x in shared[key]]
            logits, true = res[f"{key}_logits"][h][:, idx], res[f"{key}_true"][h][:, idx]
            m = label_metrics(logits, true)
            out[f"{name}/{key}"] = {"top1": m["top1_acc"], "f1": m["f1_micro"], "recall": m["recall"],
                                    "mAP": _ap(1 / (1 + np.exp(-logits)), true > 0)}
    # Food-101 held-out photos (Western dishes), scored with the run's own full vocabulary
    ds = build_dataset("food101", "val", cfg, False)
    if ds is not None:
        truth = {r["image_id"]: components(r) for r in ds.records}
        ds.transform = build_transform(cfg, False, **ck["normalization"])
        res = predict(model, loader(ds, cfg, collate, False), cfg)
        top1 = top5 = ing = 0
        for i, image_id in enumerate(res["image_id"]):
            dishes, ingredients = truth[image_id]
            order = np.argsort(-res["dishes_logits"][i])
            top = [ck["dishes"][j] for j in order[:5]]
            top1 += bool(dishes) and top[0] in dishes
            top5 += bool(set(top) & set(dishes))
            ing += ck["ingredients"][int(res["ingredients_logits"][i].argmax())] in ingredients
        n = len(res["image_id"])
        known = sum(1 for d, _ in truth.values() if set(d) & set(ck["dishes"]))
        out["food101"] = {"top1": top1 / n, "top5": top5 / n, "ingredient_top1": ing / n,
                          "dish_known": known / len(truth), "n": n}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, help="markdown report (a .json with the raw numbers is written next to it)")
    a = p.parse_args()
    runs = [r for r in a.runs if (r / "best.pt").exists()]
    vocab = [torch.load(r / "best.pt", map_location="cpu", weights_only=False) for r in runs]
    shared = {k: sorted(set.intersection(*[set(v[k]) for v in vocab]) - {"other"}) for k in ("dishes", "ingredients")}
    results = {}
    for r in runs:
        results[f"{r.parent.name}/{r.name}"] = evaluate_run(r, shared)
        print(r, json.dumps(results[f"{r.parent.name}/{r.name}"]), flush=True)

    groups = defaultdict(list)  # seeds of the same experiment together
    for k, v in results.items():
        groups[k.rsplit("_s", 1)[0]].append(v)

    def ms(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        return f"{100 * v.mean():.1f}" + (f" ± {100 * v.std(ddof=1):.1f}" if len(v) > 1 else "")

    sets = ("nutrition5k", "nutritionverse_real", "nutrition5k_side")
    lines = [f"Test sets, only the {len(shared['dishes'])} dishes / {len(shared['ingredients'])} ingredients all runs know "
             "(mean of N5k top-down, NVR, N5k side; %)", "",
             "| run | seeds | vocab (dishes / ingr.) | dish top-1 | dish mAP | ingr. top-1 | ingr. mAP | ingr. recall |",
             "|---|---|---|---|---|---|---|---|"]
    for g, vs in groups.items():
        avg = lambda key, m: [np.mean([v[f"{s}/{key}"][m] for s in sets]) for v in vs]
        lines.append(f"| {g} | {len(vs)} | {vs[0]['n_dishes']} / {vs[0]['n_ingredients']} | {ms(avg('dishes', 'top1'))} | "
                     f"{ms(avg('dishes', 'mAP'))} | {ms(avg('ingredients', 'top1'))} | {ms(avg('ingredients', 'mAP'))} | "
                     f"{ms(avg('ingredients', 'recall'))} |")
    lines += ["", "Food-101 held-out photos (101 Western dishes, %)", "",
              "| run | dish known | dish top-1 | dish top-5 | top ingredient typical |", "|---|---|---|---|---|"]
    for g, vs in groups.items():
        if "food101" in vs[0]:
            f = [v["food101"] for v in vs]
            lines.append(f"| {g} | {ms([x['dish_known'] for x in f])} | {ms([x['top1'] for x in f])} | "
                         f"{ms([x['top5'] for x in f])} | {ms([x['ingredient_top1'] for x in f])} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
        a.out.with_suffix(".json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
