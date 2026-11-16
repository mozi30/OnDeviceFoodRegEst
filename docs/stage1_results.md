# Stage 1 results: RGB food mass and nutrient estimation

Status: 2026-11-16. All runs are in `runs/`; code in `training/`.

## Summary

- **Best model: MobileNetV3-Large at 320 px**, 4.97 M parameters. It is trained on Nutrition5k overhead photos, NutritionVerse-Real and Nutrition5k side-view video frames.
- **Test score 29.1 ± 0.2** (2 seeds). The score is the mean relative error over mass, kcal, fat, carbs and protein, weighted 60 % side views / 20 % top-down / 20 % NVR.
  - Nutrition5k top-down: mass 14.7 %, kcal 20.2 %.
  - Nutrition5k side views: mass 23.6 %, kcal 31.2 %.
  - NutritionVerse-Real: mass 20.0 %, kcal 22.4 %.
- **It beats the trivial baselines by a wide margin.** The training mean scores 71–79 %, the median of the same main food 31–58 %.
- **The factorized head beats direct regression**: 25.0 vs 30.3 val score, about 4 points better on both test sets.
- **The biggest gains came from three changes:**
  1. Training on side-view frames: side-view error fell from 74–91 % to 33 %.
  2. A higher weight on the composition loss: −3.6 val points.
  3. Stable fine-tuning of the backbone: the MobileNetV4 runs had been diverging.
- **All three backbones end within ~1 point of each other.** MobileNetV3 is the smallest and the best, and the best choice for on-device use.

## 1. Task and evaluation protocol

From one RGB photo the model predicts:

- the dish's **mass** as log-mass with a variance, so every estimate comes with an uncertainty range;
- the **composition per 100 g**: protein, fat, carbs;
- **dish and ingredient labels**, as two multi-label heads.

Totals are mass × composition, and kcal = 4·protein + 4·carbs + 9·fat.

**Metric.** Relative error = MAE / mean true value × 100, per target, as in the Nutrition5k paper. A dataset's score is the mean over the five targets. The overall score is a weighted mean over the three test sets:

| Set | Weight | Val photos | Test photos (dishes) |
|---|---|---|---|
| Nutrition5k side views (video frames, 4 cameras) | 0.6 | 4,710 | 7,803 (676) |
| Nutrition5k overhead (top-down) | 0.2 | 285 | 507 (507) |
| NutritionVerse-Real (phone photos) | 0.2 | 105 | 151 (38) |

Side views carry 60 % of the weight because real photos are rarely taken straight from above. Splits are by dish, so no dish appears in more than one split. Model selection (best epoch, best configuration) uses only the val score. Test is reported, never used for choosing.

> Rounds 0–3 below were chosen with the earlier top-down-only score: the mean of Nutrition5k overhead and NVR. Side views entered the protocol for the final runs.

## 2. Final results

Side-frame configuration (§3), 40 epochs. 224 px has 3 seeds; 288 and 320 px have 2 seeds.

| Backbone | Params | Resolution | Val | **Test score** | N5k top-down | N5k side | NVR |
|---|---|---|---|---|---|---|---|
| **MobileNetV3-L** | 4.97 M | 224 | 24.4 | 30.0 ± 0.4 | 24.5 | 33.8 | 24.4 |
| | | 288 | 23.5 | 29.5 ± 0.1 | 23.6 | 33.1 | 24.4 |
| | | **320** | **23.4** | **29.1 ± 0.2** | 23.4 | 32.8 | **23.5** |
| EfficientNet-Lite0 | 4.14 M | 224 | 25.0 | 30.1 ± 0.2 | 23.5 | 33.4 | 26.6 |
| | | 288 | 24.1 | 29.9 ± 0.8 | 22.8 | 32.5 | 29.3 ± 4.2 |
| | | 320 | 24.7 | 29.4 ± 0.5 | **22.8** | **32.4** | 26.8 ± 2.5 |
| MobileNetV4-conv-M | 9.21 M | 224 | 25.0 | 30.5 ± 0.3 | 24.3 | 33.1 | 29.1 |
| | | 288 | 24.6 | 30.0 ± 0.8 | 23.4 | 32.4 | 29.4 ± 3.2 |
| | | 320 | 24.5 | 29.8 ± 0.2 | 23.8 | 32.6 | 27.5 |

What the table shows:

- **Resolution.** 224 → 320 px improves the test score by 0.7–0.9 points for every backbone. 288 and 320 px are within seed noise of each other.
- **NVR is noisy.** It has only 151 test photos from 38 dishes, and EfficientNet and MobileNetV4 vary by ±2.5–4 points between seeds there.

### 2.1 Best model in detail (MobileNetV3-L, 320 px, mean of 2 seeds)

Relative error in %, with the MAE in brackets.

| Test set | Mass | kcal | Fat | Carbs | Protein | Mean |
|---|---|---|---|---|---|---|
| N5k top-down | 14.7 (29 g) | 20.2 (52 kcal) | 28.6 (3.7 g) | 25.1 (4.9 g) | 28.0 (4.9 g) | **23.4** |
| N5k side, per frame | 23.6 (45 g) | 31.2 (80 kcal) | 44.0 (6.1 g) | 34.7 (6.7 g) | 30.7 (4.9 g) | **32.8** |
| N5k side, median over a dish's frames | 21.3 | 28.4 | 40.7 | 31.9 | 27.1 | 29.9 |
| NVR | 20.0 (112 g) | 22.4 (257 kcal) | 27.2 (15 g) | 17.9 (13 g) | 30.3 (28 g) | **23.5** |
| *Baseline: training mean*, N5k top / side / NVR | | | | | | 73.3 / 78.8 / 71.1 |
| *Baseline: median of same main food*, N5k top / side / NVR | | | | | | 54.0 / 57.6 / 31.1 |

The main-food baseline uses the true main-food label of each photo. A real system would not have that label, so this is an optimistic baseline. The model still beats it on every target and every set.

For a typical Nutrition5k plate (199 g, 255 kcal) the model is off by about 30 g and 50 kcal. Averaging over several frames of the same dish lowers the side-view error by about 3 points.

**Recognition heads**, N5k top-down test:
- Dishes: top-1 accuracy 68–70 %, micro-F1 0.56–0.60 (precision 0.85–0.90, recall 0.42–0.45).
- Ingredients: top-1 accuracy 87–88 %, micro-F1 0.51–0.52.

Without the label heads the val score is 0.1 points better, which is within noise, but test is about 1 point worse on both sets (round 1). So the heads help slightly and are needed for the later stages.

**Error by main food** (N5k top-down, mass / kcal relative error):

| Main food | Mass error | kcal error |
|---|---|---|
| Apple | 6 % | 22 % |
| Almonds | 8 % | 12 % |
| Scrambled eggs | 12 % | 9 % |
| Berries | 13 % | 19 % |
| Cauliflower | 15 % | 28 % |
| Chicken | 19 % | 25 % |

On side views, pineapple (60 % mass error) and mixed greens (34 %) are the outliers. On NVR, half a bread loaf (74 %) and meatloaf (31 %) are the outliers. Per-dish lists are in `results.json` → `test.<set>.by_primary_label`.

### 2.2 Uncertainty

The 90 % mass range is calibrated on val. The model's raw log-std is too small by a factor of about 2.8, so raw coverage is only 35–42 %. After calibration the 90 % range covers:

| N5k top-down | N5k side | NVR |
|---|---|---|
| 78–79 % | 85 % | 75–78 % |

A typical 90 % range is ÷/× 1.25 for top-down photos and ÷/× 1.36 for side views. The coverage is still below 90 % because one scale factor is fitted for all view types. Calibrating per view type (or per dataset) is an open point.

## 3. How we got there

All experiments ran on MobileNetV4-conv-M at 224 px with 2 seeds, scored with the top-down-only val score. Lower is better.

**Starting point.** Each backbone fine-tuned with default settings:

| Backbone | Val | Test N5k | Test NVR |
|---|---|---|---|
| MobileNetV3-L | 28.8 | 30.1 | 35.6 |
| EfficientNet-Lite0 | 32.4 | 31.7 | 44.6 |
| MobileNetV4-conv-M | 26.0 | 27.8 | 28.0 |

**Round 0: stability.** MobileNetV4's val loss exploded (up to NaN) shortly after the backbone was unfrozen. Three measures fixed it:
- clamp the predicted mass (at most 5 kg);
- freeze BatchNorm statistics in the backbone;
- warm up the backbone learning rate over 1 epoch after unfreezing.

Val went from 26.8 to 25.4, with no more blow-ups.

**Round 1: one change at a time** (base = 25.0):

| Change | Val | Test N5k | Test NVR |
|---|---|---|---|
| Composition-loss weight 1.0 (default 0.2) | **21.4** | 25.0 | 29.4 |
| Composition-loss weight 2.0 / 0.5 | 22.1 / 22.4 | 25.3 / 25.6 | 29.1 / 27.6 |
| 60 epochs instead of 30 | 23.3 | 25.6 | 27.0 |
| 288 px | 23.6 | 25.9 | 26.9 |
| Backbone LR = 0.3 × head LR (default 0.1 ×) | 23.6 | 25.8 | 28.1 |
| NVR sampled twice per epoch | 24.5 | 26.7 | 25.4 |

No clear effect (within ±0.5):
- EMA of the weights;
- test-time flip;
- augmentation none / strong;
- padding to square instead of cropping;
- dropout, hidden size, weight decay;
- freeze schedule.

Worse:
- **direct regression of all totals: 30.3**;
- head LR 3e-4: 33.3;
- composition weight 0.05: 34.4;
- backbone LR = 1 × head LR: 46.8, one run diverged.

**Round 2: extra data** (base 25.1):

| Variant | Val | Test N5k | Test NVR |
|---|---|---|---|
| Side-view frames, 50 % per epoch | **23.2** | 24.2 | 27.5 |
| Side-view frames, 25 % per epoch | 23.6 | **24.1** | **25.1** |
| Mix in MM-Food-100K, 10–30 % | 25.4–26.0 | 26.9–27.8 | 29.2–30.4 |
| Pretrain on MM-Food-100K, then fine-tune | 24.7–26.8 | 24.2–25.1 | 27.8–32.0 |

MM-Food-100K didn't help, mixed in or as pretraining. Its nutrition labels are LLM estimates, and the model reached only about 56–61 % error on real val after pretraining on it.

**Round 3: combining the winners** (composition weight 1.0 + backbone LR 0.3 × head LR):

| Variant | Val | Test N5k | Test NVR |
|---|---|---|---|
| 60 epochs, NVR × 2 | **19.6** | 25.1 | 27.1 |
| 60 epochs | 19.6 | 25.4 | 27.3 |
| + side frames 25 %, 40 epochs | 20.0 | **23.8** | **26.7** |
| + side frames, MM-Food pretrain | 19.9 | 24.2 | 27.8 |
| + side frames, 288 px | 20.9 | 23.5 | 28.7 |

**Switching to side-view evaluation.** The round 3 val winner (no side frames) was first trained on all three backbones. Re-scored with side views, its side-view error was 74–91 %: it predicts about 2.25× the true mass on side views. The side-frame configuration fixes that (33 % error) without hurting top-down accuracy. That made it the final configuration. It was then trained on all three backbones and at two higher resolutions (§2).

**Final configuration** (`training/experiments.py`, groups `final_side` and `final_hires`):

```
--datasets nutrition5k nutritionverse_real nutrition5k_side --sampling nutrition5k_side=0.25
--composition-weight 1.0 --backbone-lr-mult 0.3 --epochs 40 --freeze-bn --unfreeze-warmup 1
--local-images --augment basic --image-size 320 --backbone mobilenetv3_large
```

Other settings:
- AdamW with LR 1e-3 for the head and 3e-4 for the backbone, cosine decay, batch 64, AMP;
- the backbone is frozen for the first 3 epochs;
- augmentation: horizontal flip and colour jitter ±20 %, with no scale or zoom changes, because apparent size carries the mass information;
- each epoch draws a fresh random 25 % of the 40,644 side frames, about 10,000 images, next to 2,470 N5k overhead and 633 NVR photos.

## 4. Training behaviour

Runs train for a fixed number of epochs while the learning rate decays to zero. The best epoch by val score is kept. Slopes over the last 5 epochs, per epoch:

| | Train loss | Val loss | Val score |
|---|---|---|---|
| 60-epoch runs | −0.1 to −0.7 % | **+0.02 to +0.15**, rising since epoch 11–30 | −0.09 to +0.02 points |
| Side-frame runs | −1.7 to −2 % | +0.07 to +0.10 | −0.12 to +0.06 points |

- **The error has converged.** The val score is flat, and best epochs are late (36–55).
- **The val loss rises.** The model becomes overconfident: its predicted spread shrinks, so the Gaussian log-likelihood penalises each miss more. The dish and ingredient classifiers also overfit. This rise doesn't affect the error, but it is why calibration on val (§2.2) is necessary.

## 5. Caveats and open points

- **Val–test gap.** Selecting configurations by val over several rounds overfits the small val sets (285 and 105 photos). In round 3 val improved by about 3.5 points, but test by only about 1. The final comparison uses fixed configurations and several seeds to limit this.
- **NVR is small** (38 test dishes). Differences below about 2 points on NVR are not meaningful.
- **Side-view test frames are correlated.** There are about 12 frames per dish, so the effective sample size is the 676 dishes. The dish-median row in §2.1 accounts for this.
- **Training mix.** The side-view share during training (about 77 % of the images) was not tuned under the new score.
- **Resolution limit.** The cached images have a shorter side of 320 px, so higher resolutions would need re-extracting the frames.
- **Not done yet:**
  - the food-group breakdown required by the spec (e.g. soups and liquids);
  - on-device latency measurements;
  - calibration per view type.

## 6. Reproducing

```bash
uv run python -m training.prefetch convert         # N5k / NVR -> local JPEGs
uv run python -m training.prefetch side            # side-view frames (3 per video, cameras A-D)
uv run python -m training.experiments run final_side --parallel 3
uv run python -m training.experiments run final_hires --parallel 2
uv run python -m training.experiments table final_hires
uv run python -m training.reevaluate runs/final/*_s?   # re-score older checkpoints with the current protocol
```

Each run writes the following to `runs/<group>/<name>_s<seed>/`:
- `results.json`: config, all metrics and baselines;
- `summary.md`;
- `history.jsonl`: one line per epoch;
- `predictions_test.csv`: one row per test image, with the 90 % mass range;
- `labels.json`;
- `best.pt`: the checkpoint, including `log_std_scale` for the calibrated uncertainty.

Checkpoints are kept only for the `final*` groups.
