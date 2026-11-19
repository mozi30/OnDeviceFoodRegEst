"""Evaluate finished runs again with the current evaluation protocol (no training).

Used when the protocol changes, e.g. when side-view frames were added to val/test
(60 % of the score) or impossible labels were removed (``label_quality.py``). Loads
``best.pt`` of each run, recalibrates the uncertainty range on the new val sets and
rewrites ``results.json``, ``summary.md`` and ``predictions_test.csv``. The previous
``results.json`` is kept once under ``--backup`` (default ``results_topdown.json``).
The best epoch is not chosen again (it was chosen with the old val score);
``results.json`` records that under ``reevaluated``.

    uv run python -m training.reevaluate runs/final/*_s?
    uv run python -m training.reevaluate runs/final_hires/*_s? --backup results_uncleaned.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .models import FoodModel
from .trainer import (Baselines, Collate, TrainConfig, build_dataset, build_transform, calibration, evaluate, loader,
                      parse_weights, predict, std_scale, val_score, write_predictions, write_summary)


def reevaluate(run_dir: Path, workers: int, backup_name: str = "results_topdown.json") -> dict:
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    defaults = TrainConfig()
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in known})
    cfg.eval_datasets, cfg.eval_weights, cfg.workers = defaults.eval_datasets, defaults.eval_weights, workers
    cfg.clean_labels = defaults.clean_labels
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = parse_weights(cfg.eval_weights)

    dishes, ingredients, norm = ckpt["dishes"], ckpt["ingredients"], ckpt["normalization"]
    model = FoodModel(cfg.backbone, len(dishes), len(ingredients), pretrained=False,
                      hidden=cfg.hidden, dropout=cfg.dropout, mode=cfg.head_mode, rec_hidden=cfg.rec_hidden).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    collate = Collate(dishes, ingredients)
    baselines = {}
    for n in set(cfg.datasets) | set(cfg.eval_datasets):
        if (ds := build_dataset(n, "train", cfg, False)) is not None:
            baselines[n] = Baselines(ds.records)

    def run_eval(split: str, detailed: bool = False):
        parts = []
        for n in cfg.eval_datasets:
            ds = build_dataset(n, split, cfg, False)
            if ds is not None and len(ds):
                ds.transform = build_transform(cfg, False, **norm)
                parts.append(predict(model, loader(ds, cfg, collate, False), cfg))
        res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "loss"}
        return res, evaluate(res, baselines, detailed)

    val_res, val_report = run_eval("val")
    scale = std_scale(val_res["log_mass"], val_res["log_mass_std"], val_res["true"][:, 0])
    test_res, test_report = run_eval("test", detailed=True)
    test_res["log_mass_std"] = test_res["log_mass_std"] * scale
    for name, r in test_report.items():
        m = test_res["dataset"] == name
        r["calibration_raw"] = r["calibration"]
        r["calibration"] = calibration(test_res["log_mass"][m], test_res["log_mass_std"][m], test_res["true"][m, 0])

    old_path = run_dir / "results.json"
    backup = run_dir / backup_name
    if not backup.exists():
        shutil.copy(old_path, backup)
    results = json.loads(backup.read_text())
    results["config"] = dataclasses.asdict(cfg)
    results["model"]["log_std_scale"] = round(scale, 4)
    results["training"].setdefault("best_val_score_original", results["training"]["best_val_score"])
    results["training"]["best_val_score"] = round(val_score(val_report, weights), 2)
    results["val"], results["test"] = val_report, test_report
    results["test_score"] = round(val_score(test_report, weights), 2)
    results["eval_weights"] = {n: weights.get(n, 1.0) for n in test_report}
    results["reevaluated"] = f"best epoch chosen with the val score of {backup_name}"
    old_path.write_text(json.dumps(results, indent=1, default=float))
    write_predictions(run_dir / "predictions_test.csv", test_res)
    write_summary(run_dir / "summary.md", results)
    ckpt["log_std_scale"] = scale
    torch.save(ckpt, run_dir / "best.pt")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--backup", default="results_topdown.json", help="name of the copy of the previous results.json")
    a = p.parse_args()
    for run_dir in a.runs:
        r = reevaluate(run_dir, a.workers, a.backup)
        means = {n: t["photo"]["mean_rel_err_pct"] for n, t in r["test"].items()}
        print(f"{run_dir}: val {r['training']['best_val_score']} | test {r['test_score']} | {means}", flush=True)


if __name__ == "__main__":
    main()
