"""Run a group of training configurations and compare them.

    uv run python -m training.experiments run round1 --parallel 3
    uv run python -m training.experiments table round1          # Markdown table, sorted by val score

Every experiment is ``trainer`` arguments on top of ``BASE`` and is run once per
seed into ``runs/<group>/<name>_s<seed>/``. Finished runs (``results.json``
exists) are skipped, so a group can be extended and re-run. Model selection uses
only the val score (mean relative error over the five targets, weighted over the
datasets: Nutrition5k side views 60 %, Nutrition5k overhead 20 %,
NutritionVerse-Real 20 %; before ``final_side`` it was the plain mean of the two
top-down sets); the test numbers are reported but not used for choosing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

RUNS = Path("runs")
BASE = ["--backbone", "mobilenetv4_conv_medium", "--workers", "6"]
STABLE = ["--freeze-bn", "--unfreeze-warmup", "1"]  # result of round0, part of every later group
FAST = STABLE + ["--local-images"]  # from round2 on: all images from small local JPEGs (much less CPU)
SIDE = ["--datasets", "nutrition5k", "nutritionverse_real", "nutrition5k_side"]
FINAL_BACKBONES = ["mobilenetv4_conv_medium", "mobilenetv3_large", "efficientnet_lite0"]

GROUPS: dict[str, dict] = {
    # stability after unfreezing the backbone (val blow-ups in epoch 4-6 of every MobileNetV4 run)
    "round0": {"seeds": [0, 1], "experiments": {
        "base": [],
        "warmup1": ["--unfreeze-warmup", "1"],
        "freeze_bn": ["--freeze-bn"],
        "freeze_bn_warmup1": ["--freeze-bn", "--unfreeze-warmup", "1"],
    }},
    # one change at a time against the baseline
    "round1": {"seeds": [0, 1], "base": STABLE, "experiments": {
        "base": [],
        "epochs60": ["--epochs", "60"],
        "blr0.3": ["--backbone-lr-mult", "0.3"],
        "blr1": ["--backbone-lr-mult", "1.0"],
        "lr3e-4": ["--lr", "3e-4"],
        "lr3e-3": ["--lr", "3e-3"],
        "freeze1": ["--freeze-epochs", "1"],
        "freeze6": ["--freeze-epochs", "6"],
        "warmup3": ["--unfreeze-warmup", "3"],
        "ema": ["--ema", "0.995"],
        "tta": ["--tta"],
        "aug_none": ["--augment", "none"],
        "aug_strong": ["--augment", "strong"],
        "crop_pad": ["--crop", "pad"],
        "img288": ["--image-size", "288"],
        "comp0.05": ["--composition-weight", "0.05"],
        "comp0.5": ["--composition-weight", "0.5"],
        "comp1": ["--composition-weight", "1.0"],
        "comp2": ["--composition-weight", "2.0"],
        "no_label_heads": ["--dish-weight", "0", "--ingredient-weight", "0"],
        "dropout0.4": ["--dropout", "0.4"],
        "hidden1024": ["--hidden", "1024"],
        "wd0.05": ["--weight-decay", "0.05"],
        "bs32": ["--batch-size", "32"],
        "nvr_x2": ["--sampling", "nutritionverse_real=2"],
        "nvr_x3": ["--sampling", "nutritionverse_real=3"],
        "direct": ["--head-mode", "direct"],
    }},
    # pre-training on the weak MM-Food-100K labels only (evaluated on the real val sets for monitoring)
    "pretrain": {"seeds": [0], "base": FAST, "experiments": {
        "mmfood_e8": ["--datasets", "mm_food_100k", "--epochs", "8", "--freeze-epochs", "1"],
        "mmfood_side_e8": ["--datasets", "mm_food_100k", "nutrition5k_side", "--epochs", "8", "--freeze-epochs", "1",
                           "--sampling", "nutrition5k_side=0.5"],
    }},
    # more data: Nutrition5k side-angle frames (~40k, prefetched) and MM-Food-100K (30k, weak labels)
    "round2": {"seeds": [0, 1], "base": FAST, "experiments": {
        "base": [],
        "side0.1": ["--datasets", "nutrition5k", "nutritionverse_real", "nutrition5k_side", "--sampling", "nutrition5k_side=0.1"],
        "side0.25": ["--datasets", "nutrition5k", "nutritionverse_real", "nutrition5k_side", "--sampling", "nutrition5k_side=0.25"],
        "side0.5": ["--datasets", "nutrition5k", "nutritionverse_real", "nutrition5k_side", "--sampling", "nutrition5k_side=0.5"],
        "mm0.1": ["--datasets", "nutrition5k", "nutritionverse_real", "mm_food_100k", "--sampling", "mm_food_100k=0.1"],
        "mm0.3": ["--datasets", "nutrition5k", "nutritionverse_real", "mm_food_100k", "--sampling", "mm_food_100k=0.3"],
        "side0.25_mm0.1": ["--datasets", "nutrition5k", "nutritionverse_real", "nutrition5k_side", "mm_food_100k",
                           "--sampling", "nutrition5k_side=0.25", "mm_food_100k=0.1"],
        "ft_mmfood": ["--init-from", "runs/pretrain/mmfood_e8_s0/best.pt"],
        "ft_mmfood_side": ["--init-from", "runs/pretrain/mmfood_side_e8_s0/best.pt"],
        "ft_mmfood+side0.25": ["--init-from", "runs/pretrain/mmfood_e8_s0/best.pt",
                               "--datasets", "nutrition5k", "nutritionverse_real", "nutrition5k_side",
                               "--sampling", "nutrition5k_side=0.25"],
    }},
    # round-1 winners combined (composition weight 1, backbone lr x0.3, longer training) with the data options of round2
    "round3": {"seeds": [0, 1], "base": FAST + ["--composition-weight", "1.0", "--backbone-lr-mult", "0.3"], "experiments": {
        "combo_e60": ["--epochs", "60"],
        "combo_e60_nvr2": ["--epochs", "60", "--sampling", "nutritionverse_real=2"],
        "combo_side0.25_e40": SIDE + ["--epochs", "40", "--sampling", "nutrition5k_side=0.25"],
        "combo_side0.25_nvr2_e40": SIDE + ["--epochs", "40", "--sampling", "nutrition5k_side=0.25", "nutritionverse_real=2"],
        "combo_side0.25_ftmm_e40": SIDE + ["--epochs", "40", "--sampling", "nutrition5k_side=0.25",
                                          "--init-from", "runs/pretrain/mmfood_side_e8_s0/best.pt"],
        "combo_side0.25_img288_e40": SIDE + ["--epochs", "40", "--sampling", "nutrition5k_side=0.25", "--image-size", "288"],
    }},
}
# round3's side-frame config on every backbone, evaluated with side views (60 % of the val/test score)
GROUPS["final_side"] = {"seeds": [0, 1, 2], "base": GROUPS["round3"]["base"], "experiments": {
    b: GROUPS["round3"]["experiments"]["combo_side0.25_e40"] + ["--backbone", b] for b in FINAL_BACKBONES}}
# the same at higher input resolution (the local JPEGs have a shorter side of 320 px, so 320 is the maximum)
GROUPS["final_hires"] = {"seeds": [0, 1], "base": GROUPS["final_side"]["base"], "experiments": {
    f"{b}_img{size}": args + ["--image-size", str(size)]
    for b, args in GROUPS["final_side"]["experiments"].items() for size in (288, 320)}}
# pad to square instead of centre crop (side frames lose 44 % of their width with the centre crop)
GROUPS["crop_pad"] = {"seeds": [0, 1], "base": GROUPS["final_side"]["base"], "experiments": {
    "mobilenetv3_large_img320_pad": GROUPS["final_hires"]["experiments"]["mobilenetv3_large_img320"] + ["--crop", "pad"]}}
# final config trained on cleaned labels (training.label_quality, now default), centre crop and pad to square
GROUPS["clean"] = {"seeds": [0, 1], "base": GROUPS["final_side"]["base"], "experiments": {
    "mobilenetv3_large_img320": GROUPS["final_hires"]["experiments"]["mobilenetv3_large_img320"],
    "mobilenetv3_large_img320_pad": GROUPS["crop_pad"]["experiments"]["mobilenetv3_large_img320_pad"]}}


def run_one(group: str, name: str, args: list[str], seed: int, gpu_lock=None) -> str:
    run_name = f"{name}_s{seed}"
    out_dir = RUNS / group
    if (out_dir / run_name / "results.json").exists():
        return f"skip {run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "training.trainer", "--run-name", run_name, "--out-dir", str(out_dir),
           "--seed", str(seed), *BASE, *GROUPS[group].get("base", []), *args]
    t0 = time.time()
    with (out_dir / f"{run_name}.log").open("w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ | {"PYTHONWARNINGS": "ignore"})
    return f"{'done' if code == 0 else f'FAILED ({code})'} {run_name} in {(time.time() - t0) / 60:.1f} min"


def run(group: str, parallel: int, only: list[str] | None) -> None:
    spec = GROUPS[group]
    jobs = [(name, args, seed) for seed in spec["seeds"] for name, args in spec["experiments"].items()
            if not only or name in only]
    print(f"{group}: {len(jobs)} runs, {parallel} in parallel", flush=True)
    with ThreadPoolExecutor(parallel) as pool:
        for msg in pool.map(lambda j: run_one(group, *j), jobs):
            print(time.strftime("%H:%M:%S"), msg, flush=True)
    print(table(group))


def collect(group: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted((RUNS / group).glob("*/results.json")):
        name = path.parent.name.rsplit("_s", 1)[0]
        out.setdefault(name, []).append(json.loads(path.read_text()))
    return out


def _ms(values: list[float]) -> str:
    v = np.array(values, dtype=float)
    return f"{v.mean():.1f}" + (f" ± {v.std(ddof=1):.1f}" if len(v) > 1 else "")


def table(group: str) -> str:
    """Markdown table sorted by val score. Columns per test set: mean relative error (mass / kcal)."""
    sets = (("nutrition5k", "N5k top"), ("nutrition5k_side", "N5k side"), ("nutritionverse_real", "NVR"))
    rows = []
    for name, results in collect(group).items():
        val = [r["training"]["best_val_score"] for r in results]
        score = [r["test_score"] for r in results if "test_score" in r]
        cells = []
        for ds, _ in sets:
            rs = [r["test"][ds]["photo"] for r in results if ds in r["test"]]
            cells.append(f"{_ms([p['mean_rel_err_pct'] for p in rs])} ({_ms([p['mass_g']['rel_err_pct'] for p in rs])} / "
                         f"{_ms([p['kcal']['rel_err_pct'] for p in rs])})" if rs else "-")
        cov = [np.mean([t["calibration"]["coverage_90"] for t in r["test"].values()]) for r in results]
        rows.append((float(np.mean(val)), f"| {name} | {len(results)} | **{_ms(val)}** | {_ms(score) if score else '-'} | "
                     + " | ".join(cells) + f" | {_ms([100 * c for c in cov])} | "
                     f"{np.mean([r['training']['minutes'] for r in results]):.1f} |"))
    head = ("| experiment | seeds | val score | test score | " + " | ".join(f"test {t} (mass / kcal)" for _, t in sets)
            + " | 90 % cov. | min |\n|" + "---|" * (len(sets) + 6))
    return head + "\n" + "\n".join(r for _, r in sorted(rows))


def prune(group: str, keep: list[str]) -> None:
    """Delete the checkpoints of finished runs (disk space), except experiments named in ``keep``."""
    freed = 0
    for path in (RUNS / group).glob("*/best.pt"):
        if (path.parent / "results.json").exists() and path.parent.name.rsplit("_s", 1)[0] not in keep:
            freed += path.stat().st_size
            path.unlink()
    print(f"{group}: freed {freed / 2**20:.0f} MB")


def final(group: str, parallel: int, seeds: list[int]) -> None:
    """Train the best experiment of ``group`` (by mean val score) on every backbone of FINAL_BACKBONES."""
    scores = {name: np.mean([r["training"]["best_val_score"] for r in res]) for name, res in collect(group).items()}
    best = min(scores, key=scores.get)
    args = GROUPS[group].get("base", []) + GROUPS[group]["experiments"][best]
    print(f"final: best of {group} is {best} (val {scores[best]:.2f}) -> {args}", flush=True)
    experiments = {}
    for backbone in FINAL_BACKBONES:
        experiments[f"{backbone}"] = args + ["--backbone", backbone]
    GROUPS["final"] = {"seeds": seeds, "experiments": experiments, "source": f"{group}/{best}"}
    (RUNS / "final").mkdir(parents=True, exist_ok=True)
    (RUNS / "final" / "config.json").write_text(json.dumps({"source": f"{group}/{best}", "args": args}, indent=1))
    run("final", parallel, None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=["run", "table", "prune", "final"])
    p.add_argument("group", choices=sorted(GROUPS) + ["final"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--only", nargs="*")
    p.add_argument("--keep", nargs="*", default=[], help="prune: experiments whose checkpoints are kept")
    a = p.parse_args()
    if a.action == "run":
        run(a.group, a.parallel, a.only)
    elif a.action == "final":
        final(a.group, a.parallel, a.seeds)
    elif a.action == "prune":
        prune(a.group, a.keep)
    else:
        print(table(a.group))


if __name__ == "__main__":
    main()
