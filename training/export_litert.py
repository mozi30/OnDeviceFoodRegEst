"""Export a trained Stage 1 checkpoint to LiteRT (.tflite) for the Android app.

litert-torch needs an older torch than the training environment, so the export runs
in its own virtual environment (CPU only):

    uv venv -p 3.12 ~/.cache/ondevice_food/export-venv
    VIRTUAL_ENV=~/.cache/ondevice_food/export-venv uv pip install "torch==2.14.*" "torchvision==0.29.*" \
        --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=~/.cache/ondevice_food/export-venv uv pip install "litert-torch==0.10.*" "timm>=1.0.31" numpy pillow
    PYTHONPATH=. ~/.cache/ondevice_food/export-venv/bin/python -m training.export_litert \
        runs/final_hires/mobilenetv3_large_img320_s0/best.pt --out export/food_stage1

Model contract (also written to ``<out>.json`` for the app):

- input  ``image``: float32 [1, H, W, 3], RGB in 0..1, H = W = ``image_size``; the app
  resizes the shorter side to H and centre-crops, exactly like the val/test transform.
  ImageNet normalisation happens inside the model.
- output: one float32 vector [1, 7 + n_dishes + n_ingredients]
  ``[mass_g, kcal, fat_g, carb_g, protein_g, log_mass, log_std90, dish probs..., ingredient probs...]``
  where ``log_std90`` is the calibrated log-std; the 90 % mass range is
  ``exp(log_mass ± 1.645 * log_std90)``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .models import FoodModel

Z90 = 1.645


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: FoodModel, mean, std, log_std_scale: float):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.log_std_scale = float(log_std_scale)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image.permute(0, 3, 1, 2) - self.mean) / self.std
        out = self.model(x)
        log_std = (0.5 * out["log_mass_var"].clamp(-7, 7)).exp() * self.log_std_scale
        return torch.cat([out["targets"], out["log_mass"].unsqueeze(-1), log_std.unsqueeze(-1),
                          torch.sigmoid(out["dish_logits"]), torch.sigmoid(out["ingredient_logits"])], dim=-1)


def load(checkpoint: Path) -> tuple[ExportWrapper, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = FoodModel(cfg["backbone"], len(ckpt["dishes"]), len(ckpt["ingredients"]), pretrained=False,
                      hidden=cfg["hidden"], dropout=cfg["dropout"], mode=cfg["head_mode"], rec_hidden=cfg.get("rec_hidden", 0))
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = ckpt["normalization"]
    wrapper = ExportWrapper(model, norm["mean"], norm["std"], ckpt.get("log_std_scale", 1.0)).eval()
    meta = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint),
        "backbone": cfg["backbone"],
        "image_size": cfg["image_size"],
        "crop": cfg.get("crop", "center"),
        "pad_fill": [124, 116, 104],  # trainer.PadSquare
        "input": ("float32 [1, H, W, 3] RGB 0..1, shorter side resized to H then centre crop"
                  if cfg.get("crop", "center") == "center" else
                  "float32 [1, H, W, 3] RGB 0..1, padded to a square with pad_fill (image centred), then resized to H x H"),
        "output_layout": ["mass_g", "kcal", "fat_g", "carb_g", "protein_g", "log_mass", "log_std90"]
                         + ["dish:" + d for d in ckpt["dishes"]] + ["ingredient:" + i for i in ckpt["ingredients"]],
        "n_fixed": 7,
        "dishes": ckpt["dishes"],
        "ingredients": ckpt["ingredients"],
        "z90": Z90,
        "log_std_scale": ckpt.get("log_std_scale", 1.0),
        "kcal_per_g": {"protein": 4, "fat": 9, "carb": 4},
        "epoch": ckpt["epoch"],
    }
    return wrapper, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=Path("export/food_stage1"))
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                   help="also write <out>_fp16.tflite (float16 weights)")
    a = p.parse_args()
    import litert_torch

    wrapper, meta = load(a.checkpoint)
    size = meta["image_size"]
    sample = torch.rand(1, size, size, 3)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    edge = litert_torch.convert(wrapper, (sample,))
    fp32_path = a.out.with_name(a.out.name + "_fp32.tflite")
    edge.export(str(fp32_path))
    files = {"fp32": fp32_path.name}

    # check: LiteRT output == PyTorch output
    with torch.no_grad():
        ref = wrapper(sample).numpy()
    got = np.asarray(edge(sample))
    err = float(np.abs(got - ref).max())
    print(f"fp32: {fp32_path} ({fp32_path.stat().st_size / 2**20:.1f} MB), max |diff| vs torch {err:.2e}")
    checks = {"fp32_max_abs_diff": err}

    if a.fp16:
        from ai_edge_quantizer import qtyping, quantizer, recipe, recipe_manager
        rp = recipe_manager.RecipeManager()  # float16 weights, float compute (no fp16 preset in ai-edge-quantizer)
        rp.add_quantization_config(
            regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
            op_config=qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                compute_precision=qtyping.ComputePrecision.FLOAT, explicit_dequantize=True),
            algorithm_key=recipe.AlgorithmName.FLOAT_CASTING)
        qt = quantizer.Quantizer(str(fp32_path))
        qt.load_quantization_recipe(rp.get_quantization_recipe())
        fp16_path = a.out.with_name(a.out.name + "_fp16.tflite")
        qt.quantize().export_model(str(fp16_path), overwrite=True)
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=str(fp16_path))
        it.allocate_tensors()
        it.set_tensor(it.get_input_details()[0]["index"], sample.numpy())
        it.invoke()
        got16 = it.get_tensor(it.get_output_details()[0]["index"])
        rel = float(np.abs(got16[0, :5] - ref[0, :5]).max() / np.abs(ref[0, :5]).max())
        print(f"fp16: {fp16_path} ({fp16_path.stat().st_size / 2**20:.1f} MB), max rel. diff of the totals {rel:.2e}")
        files["fp16"] = fp16_path.name
        checks["fp16_max_rel_diff_totals"] = rel

    meta["files"], meta["checks"] = files, checks
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"metadata: {a.out.with_suffix('.json')}  (output length {len(meta['output_layout'])}, "
          f"90 % factor at log_std 1: {math.exp(Z90):.2f})")


if __name__ == "__main__":
    main()
