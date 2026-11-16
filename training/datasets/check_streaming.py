"""Smoke test: stream a few samples from every dataset and report the throughput.

    uv run training/datasets/check_streaming.py [-n 32] [--workers 4]
"""

import argparse
import logging
import time

from torch.utils.data import DataLoader
from torchvision import transforms

if not __package__:  # run as a file: make the repository root importable
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.datasets import (  # noqa: E402
    MMFood100K, Nutrition5k, Nutrition5kSideAngles, NutritionVerseReal, collate_samples)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=32, help="samples per dataset")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    common = dict(transform=tf, shuffle=True, draft_size=256)
    for name, make in [
        ("nutrition5k", lambda: Nutrition5k("train", **common)),
        ("nutrition5k+depth", lambda: Nutrition5k("train", depth=True, **common)),
        ("nutrition5k_side", lambda: Nutrition5kSideAngles("train", frames_per_video=4, **common)),
        ("nutritionverse_real", lambda: NutritionVerseReal("train", **common)),
        ("mm_food_100k", lambda: MMFood100K("train", **common)),
    ]:
        t0 = time.perf_counter()
        ds = make()
        t1 = time.perf_counter()
        loader = DataLoader(ds, batch_size=8, num_workers=args.workers, collate_fn=collate_samples)
        seen, last = 0, None
        for batch in loader:
            seen, last = seen + len(batch["image_id"]), batch
            if seen >= args.n:
                break
        dt = time.perf_counter() - t1
        print(f"{name:20s} {len(ds):6d} records | index {t1 - t0:5.1f}s | {seen} samples in {dt:5.1f}s "
              f"({seen / dt:5.1f} img/s) | image {tuple(last['image'].shape)} | "
              f"targets[0] {[round(x, 1) for x in last['targets'][0].tolist()]}")


if __name__ == "__main__":
    main()
