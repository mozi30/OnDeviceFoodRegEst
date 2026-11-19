# Stage 1 results: RGB food mass and nutrient estimation

Status: 2026-11-19, end of part 1. All runs are in `runs/`; code in `training/`. The exported model is in the app repository (`OnDeviceFoodRegEstApp`, default model).

## Summary

- **Final model: MobileNetV3-Large at 320 px, pad to square, own recognition layer** (8.4 M parameters, 16 MB as FP16 LiteRT). Weight and nutrition are learned only from measured data (Nutrition5k overhead and side-view frames, NutritionVerse-Real); dishes and ingredients also learn from MM-Food-100K and Food-101.
- **Test score 23.5 ± 0.1** (2 seeds). The score is the mean relative error over mass, kcal, fat, carbs and protein, weighted 60 % side views / 20 % top-down / 20 % NVR.
  - Nutrition5k top-down: mass 13.9 %, kcal 18.8 %.
  - Nutrition5k side views: mass 15.8 %, kcal 20.4 %.
  - NutritionVerse-Real: mass 20.0 %, kcal 21.7 %.
- **Recognition:** the right dish comes first in 81 % of the test photos and the right ingredient in 89 % (first model: 74 % / 84 %). The model knows about 1,300 dishes and 860 ingredients, including the 101 mostly Western Food-101 dishes: 54 % right at the first guess, 79 % in the top 5.
- **It beats the trivial baselines by a wide margin.** The training mean scores 71–76 %, the median of the same main food 31–54 %.
- **Steps since the first write-up (test score 29.1):**
  1. Removing physically impossible labels from the test sets: 29.1 → 25.0; retraining on clean labels: 24.6. Two mislabelled side-view test dishes alone had inflated the side-view error.
  2. Pad to square instead of centre crop: 24.6 → 24.1 (both retrained on clean labels).
  3. Own recognition layer + recognition-only data (MM-Food-100K, Food-101): 24.1 → 23.5, and much better recognition.
- **Most of the remaining error is in fat and protein** (26–31 %); mass is at 14–20 %.

## Progression of the test score

MobileNetV3-L, 320 px, 2 seeds each, all scored with the current protocol (clean test labels).

| Model | Val | **Test score** | N5k top-down | N5k side | NVR |
|---|---|---|---|---|---|
| First write-up, uncleaned training, centre crop (re-scored) | 20.3 | 25.0 ± 0.2 | 23.3 | 25.9 | 23.6 |
| Clean training, centre crop | 20.3 | 24.6 ± 0.1 | 23.0 | 25.8 | 22.6 |
| Clean training, pad to square | 19.9 | 24.1 ± 0.3 | 22.7 | 24.6 | 23.9 |
| + MM-Food-100K (recognition only), shared head | 19.1 | 24.0 ± 0.3 | 22.4 | 24.1 | 25.1 |
| + own 1024 recognition layer | 19.2 | 23.6 ± 0.4 | 22.2 | 24.1 | 23.7 |
| **+ Food-101 with typical ingredients (final)** | **18.9** | **23.5 ± 0.1** | 22.3 | **23.9** | **23.5** |

The first write-up reported 29.1 for the same model; that number was scored on test sets that still contained impossible labels (§3.1).

## 1. Task and evaluation protocol

From one RGB photo the model predicts:

- the dish's **mass** as log-mass with a variance, so every estimate comes with an uncertainty range;
- the **composition per 100 g**: protein, fat, carbs;
- **dish and ingredient labels**, as two multi-label heads (with their own hidden layer in the final model).

Totals are mass × composition, and kcal = 4·protein + 4·carbs + 9·fat.

**Metric.** Relative error = MAE / mean true value × 100, per target, as in the Nutrition5k paper. A dataset's score is the mean over the five targets. The overall score is a weighted mean over the three test sets:

| Set | Weight | Val photos | Test photos (dishes) |
|---|---|---|---|
| Nutrition5k side views (video frames, 4 cameras) | 0.6 | 4,698 | 7,758 (672) |
| Nutrition5k overhead (top-down) | 0.2 | 285 | 506 (506) |
| NutritionVerse-Real (phone photos) | 0.2 | 105 | 151 (38) |

Side views carry 60 % of the weight because real photos are rarely taken straight from above. Splits are by dish, so no dish appears in more than one split. Model selection (best epoch, best configuration) uses only the val score. Test is reported, never used for choosing.

Photo counts are after label cleaning (§3.1). Weight, kcal and macros are only ever trained and scored on measured labels (Nutrition5k, NutritionVerse-Real); MM-Food-100K and Food-101 train only the recognition heads.

> Rounds 0–3 (§4.1) were chosen with the earlier top-down-only score: the mean of Nutrition5k overhead and NVR. Side views entered the protocol for the final runs.

## 2. Final model in detail

`runs/western/mm_f101_s{0,1}`, configuration in `training/experiments.py`, group `western`. The app uses seed 1 (better val score); its FP16 LiteRT export gives exactly the PyTorch test score (23.56) on the desktop.

### 2.1 Weight and nutrition (mean of 2 seeds)

Relative error in %, with the MAE in brackets.

| Test set | Mass | kcal | Fat | Carbs | Protein | Mean |
|---|---|---|---|---|---|---|
| N5k top-down | 13.9 (28 g) | 18.8 (48 kcal) | 28.0 (4 g) | 24.7 (5 g) | 26.0 (5 g) | **22.3** |
| N5k side, per frame | 15.8 (28 g) | 20.4 (46 kcal) | 29.7 (3 g) | 26.2 (5 g) | 27.4 (4 g) | **23.9** |
| N5k side, median over a dish's frames | 13.8 | 18.1 | 26.5 | 23.3 | 24.3 | 21.2 |
| NVR | 20.0 (112 g) | 21.7 (250 kcal) | 27.8 (16 g) | 16.4 (12 g) | 31.3 (29 g) | **23.5** |
| *Baseline: training mean*, N5k top / side / NVR | | | | | | 73.0 / 76.4 / 71.1 |
| *Baseline: median of same main food*, N5k top / side / NVR | | | | | | 54.0 / 53.4 / 31.1 |

The main-food baseline uses the true main-food label of each photo, so it is optimistic. The model beats both baselines on every target and every set.

For a typical Nutrition5k plate (199 g, 255 kcal) the model is off by about 28 g and 47 kcal. Side views are now almost as good as top-down photos; averaging over several frames of a dish lowers the side-view error by another 2.7 points.

**Error by main food** (mass / kcal relative error, seed 1):

| Main food | N5k top-down | N5k side |
|---|---|---|
| Scrambled eggs | 9 / 8 % | 13 / 14 % |
| Cauliflower | 10 / 25 % | 16 / 28 % |
| Almonds | 11 / 15 % | 15 / 15 % |
| Berries | 14 / 21 % | 17 / 27 % |
| Cheese pizza | 14 / 12 % | 16 / 16 % |
| Chicken | 16 / 20 % | 18 / 21 % |

On NVR, half a bread loaf (51 % mass error) and meatloaf (32 %) are the outliers. Per-dish lists are in `results.json` → `test.<set>.by_primary_label`.

### 2.2 Recognition

Models trained with extra recognition data know many more names, so the plain test metrics are not comparable. `training/eval_recognition.py` scores every model only on the 49 dishes and 170 ingredients that all compared models know (mean over the three test sets, 2 seeds each):

| Model | Vocabulary (dishes / ingr.) | Dish top-1 | Dish mAP | Ingr. top-1 | Ingr. mAP | Ingr. recall |
|---|---|---|---|---|---|---|
| Clean, pad to square | 49 / 170 | 74.3 | 49.5 | 83.6 | 44.0 | 40.0 |
| + MM-Food-100K, shared head | 673 / 634 | 68.2 | 47.0 | 78.4 | 40.6 | 30.0 |
| + own 1024 recognition layer | 673 / 634 | 83.5 | 59.5 | 86.3 | 53.6 | 55.5 |
| Own 1024 layer, *no* MM-Food | 49 / 170 | 81.6 | 60.5 | 88.6 | 54.0 | 57.0 |
| **Final (MM-Food + Food-101, own layer)** | 1,300 / 862 | 80.8 | 61.3 | 88.8 | 51.4 | 57.6 |

Top-1: the highest-scoring label is one of the true labels. mAP: how high the true labels rank, independent of the 0.5 threshold. Recall: share of the true labels above the threshold.

- **The shared 512 hidden layer was the bottleneck.** With MM-Food on the shared layer, recognition got *worse* (the model became more cautious: recall 40 → 30 %). With its own 1024 layer, recognition improves clearly, and weight and nutrition improve slightly too.
- **On these test labels the gain comes from the layer, not the data**: the own layer without MM-Food is just as good. The extra data adds the ~1,250 dish names the test sets cannot measure, without hurting the measured ones.
- **MM-Food's ingredient lists are incomplete** (3.5 names per photo, 1.85 of the shared ingredients, vs 5–7 for Nutrition5k). Ignoring its unnamed ingredients (`--weak-ingredient-negatives 0`) does not help on its own and only slightly on top of the own layer.

**Western dishes** (2,526 held-out Food-101 photos, all 101 dishes, ~25 each):

| Model | Dish names known | Right dish first | Right dish in top 5 |
|---|---|---|---|
| Clean, pad to square | 8 % | 1 % | 3 % |
| MM-Food + own layer | 29 % | 3 % | 12 % |
| **Final** | **98 %** | **54 %** | **79 %** |

These photos come from the same source as Food-101's training photos, so real (e.g. Austrian) meal photos will likely score lower. Steak and chicken wings exist as ingredients in the vocabulary, so those two Food-101 dishes count as unknown dish names.

### 2.3 Uncertainty

The 90 % mass range is calibrated on val (raw log-std × 3.06). The calibrated 90 % range covers:

| N5k top-down | N5k side | NVR |
|---|---|---|
| 75 % | 85 % | 81 % |

A typical 90 % range is ÷/× 1.24 for top-down photos and ÷/× 1.31 for side views and NVR. Coverage is still below 90 % because one scale factor is fitted for all view types; calibrating per view type is an open point.

## 3. Changes after the first write-up

### 3.1 Label cleaning

Side views scored much worse than top-down photos (32.8 vs 23.4). Looking at the worst side-view dishes showed impossible labels: e.g. 172 g of macros per 100 g, 5,966 kcal per 100 g, 1 g or 7,974 g dishes. `training/label_quality.py` now drops records with

- more than 100 g of macros per 100 g, or more than 900 kcal per 100 g;
- a mass below 5 g or above 3,000 g;
- only "plate only" as ingredient.

This removed 15 dishes overall (in test: 1 top-down dish, 4 side-view dishes with 45 frames). Two mislabelled side-view test dishes alone had raised the side-view error from 26.3 to 32.9. Re-scoring the first model on the clean test sets (`training/reevaluate.py`) gives 25.0 instead of 29.1; retraining on clean labels 24.6. Cleaning is on by default (`--clean-labels`).

### 3.2 Pad to square

Centre cropping cut off the plate edges on the 4:3 photos and side-view frames. Padding the whole photo to a square with the ImageNet mean colour (RGB 124, 116, 104) and resizing it keeps the whole plate. Retrained on clean labels: 24.1 ± 0.3 vs 24.6 ± 0.1 for the centre crop, mainly on side views (24.6 vs 25.8). `training/visualize.py` shows training samples with both crops.

### 3.3 Recognition-only data and the own recognition layer

- **MM-Food-100K** (95,811 usable photos, LLM-generated labels) and **Food-101** (75,750 photos, 101 dishes, dish label only) are used as *weak* datasets (`--weak-datasets`): they train only the dish and ingredient heads. Their nutrition labels are never used, neither for weight nor for kcal or macros.
- MM-Food free-text names are mapped to the common vocabulary by name (`labels.mm_food_components`: singular/plural, aliases, dish vs ingredient by the known map).
- Food-101 photos get the typical ingredients of their dish from a hand-written list (`training/datasets/food101_ingredients.py`, e.g. spaghetti bolognese → spaghetti, pasta, ground beef, tomato sauce, onions, parmesan cheese). They count only as present, never as absent, because a given plate may differ.
- A name from the weak datasets enters the vocabulary from 10 photos (`--min-label-count-weak`); names from measured datasets from 5 dishes.
- The recognition heads get their own hidden layer (`--rec-hidden 1024`) instead of sharing the 512 layer with weight and nutrition. It adds about 1–2 MB and well under 1 ms on the phone.
- Sampling per epoch: 25 % of the side frames, 20 % of MM-Food, 20 % of Food-101, plus all Nutrition5k overhead and NVR photos.

Final configuration (`training/experiments.py`, group `western`):

```
--datasets nutrition5k nutritionverse_real nutrition5k_side mm_food_100k food101
--sampling nutrition5k_side=0.25 mm_food_100k=0.2 food101=0.2
--crop pad --rec-hidden 1024 --weak-ingredient-negatives 0.0 --min-label-count-weak 10
--composition-weight 1.0 --backbone-lr-mult 0.3 --epochs 40 --freeze-bn --unfreeze-warmup 1
--local-images --augment basic --image-size 320 --backbone mobilenetv3_large
```

Other settings:
- AdamW with LR 1e-3 for the head and 3e-4 for the backbone, cosine decay, batch 64, AMP;
- the backbone is frozen for the first 3 epochs;
- augmentation: horizontal flip and colour jitter ±20 %, with no scale or zoom changes, because apparent size carries the mass information.

## 4. How the training recipe was found

The training recipe was found before the label cleaning and with centre crops; it was kept unchanged for the later steps.

### 4.1 Rounds 0–3: training recipe (uncleaned labels)

These experiments ran on MobileNetV4-conv-M at 224 px with 2 seeds, scored with the top-down-only val score. Lower is better.

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

### 4.2 Backbones and resolution (uncleaned labels)

Side-frame configuration (§4.1), 40 epochs, centre crop, scored on the uncleaned test sets (these are the numbers of the first write-up). 224 px has 3 seeds; 288 and 320 px have 2 seeds.

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

## 5. Training behaviour

Runs train for a fixed number of epochs while the learning rate decays to zero. The best epoch by val score is kept. Slopes over the last 5 epochs, per epoch:

| | Train loss | Val loss | Val score |
|---|---|---|---|
| 60-epoch runs | −0.1 to −0.7 % | **+0.02 to +0.15**, rising since epoch 11–30 | −0.09 to +0.02 points |
| Side-frame runs | −1.7 to −2 % | +0.07 to +0.10 | −0.12 to +0.06 points |

- **The error has converged.** The val score is flat, and best epochs are late (36–55).
- **The val loss rises.** The model becomes overconfident: its predicted spread shrinks, so the Gaussian log-likelihood penalises each miss more. The dish and ingredient classifiers also overfit. This rise doesn't affect the error, but it is why calibration on val (§2.3) is necessary.

## 6. Caveats and open points

- **Val–test gap.** Selecting configurations by val over several rounds overfits the small val sets (285 and 105 photos). The final comparisons use fixed configurations and 2 seeds each to limit this.
- **NVR is small** (38 test dishes). Differences below about 2 points on NVR are not meaningful.
- **Side-view test frames are correlated.** There are about 12 frames per dish, so the effective sample size is the 672 dishes. The dish-median row in §2.1 accounts for this.
- **Recognition is only partly measured.** The test sets cover 49 dishes; the Food-101 held-out photos are from the training distribution. Real meal photos (own test set) are needed to judge the ~1,300 names.
- **Weak label quality.** MM-Food names contain quirks (fruits as dishes, the same name as dish and ingredient, vague names such as "sauce"); they were not cleaned by hand.
- **Resolution limit.** The cached images have a shorter side of 320 px, so higher resolutions would need re-extracting the frames.
- **Not done yet:**
  - the food-group breakdown required by the spec (e.g. soups and liquids);
  - on-device latency of the final model (desktop CPU: 2 ms per image; the first models ran on the Pixel GPU);
  - calibration per view type.

## 7. Reproducing

```bash
uv run python -m training.prefetch convert           # N5k / NVR -> local JPEGs
uv run python -m training.prefetch side              # side-view frames (3 per video, cameras A-D)
uv run python -m training.prefetch mmfood --n-train 100000 --n-val 2000   # all usable photos
uv run --with pyarrow python -m training.prefetch food101
uv run python -m training.experiments run western --parallel 1
uv run python -m training.experiments table western
uv run python -m training.eval_recognition runs/clean/*_pad_s? runs/western/mm_f101_s? --out runs/recognition.md
uv run python -m training.reevaluate runs/final_hires/*_s? --backup results_uncleaned.json   # re-score older runs
```

Export for the app (in the export venv, see `training/export_litert.py`):

```bash
PYTHONPATH=. ~/.cache/ondevice_food/export-venv/bin/python -m training.export_litert runs/western/mm_f101_s1/best.pt --out export/food_stage1_western
PYTHONPATH=. ~/.cache/ondevice_food/export-venv/bin/python -m training.eval_litert export/food_stage1_western_fp16.tflite --crop pad
```

Disk use: local images about 9 GB (MM-Food 4.3 GB, Food-101 2.6 GB, side frames 2.1 GB, N5k/NVR 0.1 GB).

Each run writes the following to `runs/<group>/<name>_s<seed>/`:
- `results.json`: config, all metrics and baselines;
- `summary.md`;
- `history.jsonl`: one line per epoch;
- `predictions_test.csv`: one row per test image, with the 90 % mass range;
- `labels.json`;
- `best.pt`: the checkpoint, including `log_std_scale` for the calibrated uncertainty.
