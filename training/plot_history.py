"""Training curves of several runs as one self-contained, interactive HTML page.

Small multiples (one shared epoch axis each): train loss, val loss, val score and the
val score per evaluation set. Runs are grouped by a label (same colour); seeds of a
group differ by line style.

    uv run python -m training.plot_history \
        "center crop=runs/final_hires/mobilenetv3_large_img320_s?" \
        "pad to square=runs/crop_pad/mobilenetv3_large_img320_pad_s?" \
        --out runs/crop_pad/curves.html

The training loop logs no training *score* (only the loss), so there is no train-score panel.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

SHORT = {"nutrition5k": "N5k overhead", "nutrition5k_side": "N5k side views", "nutritionverse_real": "NutritionVerse-Real"}


def load(spec: str) -> dict:
    label, _, pattern = spec.partition("=")
    if not pattern:
        raise SystemExit(f"expected <label>=<run dir glob>, got {spec!r}")
    runs = []
    for d in sorted(glob.glob(pattern)):
        hist = Path(d) / "history.jsonl"
        if not hist.exists():
            continue
        rows = [json.loads(line) for line in hist.open()]
        runs.append({
            "name": Path(d).name,
            "done": (Path(d) / "results.json").exists(),
            "epoch": [r["epoch"] for r in rows],
            "train_loss": [r["train_loss"]["total"] for r in rows],
            "val_loss": [r["val_loss"] for r in rows],
            "val_score": [r["val_score"] for r in rows],
            "per_set": {k: [r["val"].get(k, {}).get("mean_rel_err_pct") for r in rows] for k in rows[0]["val"]},
            "frozen": sum(1 for r in rows if r.get("backbone_frozen")),
        })
    if not runs:
        raise SystemExit(f"no history.jsonl matches {pattern!r}")
    return {"label": label, "runs": runs}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="+", help='"<label>=<glob of run dirs>"')
    p.add_argument("--out", type=Path, default=Path("runs/curves.html"))
    p.add_argument("--title", default="Training curves")
    p.add_argument("--note", default="", help="extra sentence under the title")
    a = p.parse_args()
    data = [load(g) for g in a.groups]
    html = (Path(__file__).with_name("plot_history.html").read_text()
            .replace("__TITLE__", a.title).replace("__NOTE__", a.note).replace("__DATA__", json.dumps({"groups": data, "short": SHORT})))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
