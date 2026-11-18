"""Evaluate an exported LiteRT model on the local test sets (same metric as the trainer).

Runs in the export environment (see ``export_litert.py``) and needs the local JPEGs
from ``training.prefetch``. Used to check that the export (FP32 / FP16 / INT8) keeps
the accuracy of the PyTorch checkpoint.

    PYTHONPATH=. ~/.cache/ondevice_food/export-venv/bin/python -m training.eval_litert \
        export/food_stage1_fp16.tflite --compare runs/final_hires/mobilenetv3_large_img320_s0/results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

from .label_quality import label_problem

LOCAL_DIR = Path.home() / ".cache" / "ondevice_food" / "local"
TARGET_KEYS = ("mass_g", "kcal", "fat_g", "carb_g", "protein_g")
WEIGHTS = {"nutrition5k": 0.2, "nutritionverse_real": 0.2, "nutrition5k_side": 0.6}


def records(name: str, split: str) -> list[dict]:
    recs = [json.loads(line) for line in (LOCAL_DIR / name / f"records_{split}.jsonl").open()]
    return [r for r in recs if label_problem(r) is None]  # same cleaning as the trainer


def pad_square(img: Image.Image, fill=(124, 116, 104)) -> Image.Image:
    """Same as trainer.PadSquare (the trainer imports torch/timm/datasets, not available here)."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def rel_errors(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    out = {}
    for i, k in enumerate(TARGET_KEYS):
        m = np.isfinite(true[:, i])
        out[k] = float(np.abs(pred[m, i] - true[m, i]).mean() / true[m, i].mean() * 100)
    out["mean"] = float(np.mean([out[k] for k in TARGET_KEYS]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model", type=Path)
    p.add_argument("--compare", type=Path, help="results.json of the PyTorch run")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--crop", choices=["center", "pad"], default="center", help="preprocessing the model was trained with")
    a = p.parse_args()
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(a.model), num_threads=a.threads)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    size = int(inp["shape"][1])
    if a.crop == "pad":
        tf = transforms.Compose([pad_square, transforms.Resize((size, size))])
    else:
        tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size)])
    ref = json.loads(a.compare.read_text())["test"] if a.compare else {}

    scores, times = {}, []
    for name in WEIGHTS:
        recs = records(name, "test")
        pred, true = [], []
        for r in recs:
            x = np.asarray(tf(Image.open(r["_path"]).convert("RGB")), dtype=np.float32)[None] / 255.0
            t0 = time.perf_counter()
            it.set_tensor(inp["index"], x)
            it.invoke()
            times.append(time.perf_counter() - t0)
            pred.append(it.get_tensor(out["index"])[0, :5])
            true.append([r.get(k, np.nan) if r.get(k) is not None else np.nan for k in TARGET_KEYS])
        e = rel_errors(np.array(pred), np.array(true, dtype=np.float64))
        scores[name] = e["mean"]
        line = f"{name:20s} n={len(recs):5d} | mean {e['mean']:.2f} | mass {e['mass_g']:.2f} | kcal {e['kcal']:.2f}"
        if name in ref:
            line += f" || torch mean {ref[name]['photo']['mean_rel_err_pct']:.2f}"
        print(line, flush=True)
    total = sum(WEIGHTS[n] * s for n, s in scores.items())
    print(f"test score {total:.2f} | {a.model.name} {a.model.stat().st_size / 2**20:.1f} MB | "
          f"desktop CPU {1000 * np.median(times):.1f} ms/image ({a.threads} threads)")


if __name__ == "__main__":
    main()
