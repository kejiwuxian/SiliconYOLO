#!/usr/bin/env python
"""Silicon YOLO — BONUS: early-conv feature-map montage ("the chip thinking").

Captures the INT8 output activations of an early Conv2d layer (default the stem,
model.0.conv) on one real COCO image while the model runs the quantized datapath,
and tiles the channels into a montage. This visualizes the intermediate INT8
feature maps the accelerator streams between layers.

Output: artifacts/sim_layer_activations.png

Usage:
  python sim/run_layer_activations.py
  python sim/run_layer_activations.py --layer model.1.conv --image-index 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

SCALES = ROOT / "hwconst" / "quant_scales.json"
ARTIFACTS = Path(r"C:\Users\light\AppData\Roaming\simular-unified-ui\SimularFiles\artifacts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="model.0.conv", help="Conv2d layer to visualize")
    ap.add_argument("--image-index", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-channels", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(ARTIFACTS))
    ap.add_argument("--scales", default=str(SCALES))
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image, ImageDraw, ImageFont

    from common import calib_images, load_detection_model, torch_device

    # reuse the verified INT8 datapath builders from the detection sim
    sys.path.insert(0, str(ROOT / "sim"))
    from run_fixedpoint_sim import _attach_act_quant, _fake_quant_weights

    layers = json.loads(Path(args.scales).read_text())["layers"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch_device(args.device)

    model = load_detection_model().to(device)
    model.fuse()
    model.eval()
    _fake_quant_weights(model, layers)
    qhandles = _attach_act_quant(model, layers)

    # capture the chosen layer's INT8 output (quant hook already applied → values
    # are on the INT8 grid; we recover the integer codes by dividing by the scale)
    act_scale = float(layers[args.layer]["activation"]["scale"])
    captured = {}

    def cap_hook(mod, inp, out):
        t = out[0] if isinstance(out, (list, tuple)) else out
        captured["fmap"] = t.detach().cpu().numpy()[0]  # (C,H,W) dequantized INT8

    target = dict(model.named_modules())[args.layer]
    h = target.register_forward_hook(cap_hook)

    # run one image
    imgs = list(calib_images(args.image_index + 1, args.imgsz))
    x, _meta = imgs[args.image_index]
    with torch.no_grad():
        model(x.unsqueeze(0).to(device))
    h.remove()
    for q in qhandles:
        q.remove()

    fmap = captured["fmap"]  # (C,H,W)
    codes = np.rint(fmap / act_scale).astype(np.int32)  # INT8 integer codes
    C, H, W = codes.shape
    nshow = min(args.max_channels, C)
    cols = 4
    rows = (nshow + cols - 1) // cols
    cell = 150
    pad = 10
    label_h = 26
    grid_w = cols * cell + (cols + 1) * pad
    grid_h = rows * (cell + label_h) + (rows + 1) * pad + 70

    canvas = Image.new("RGB", (grid_w, grid_h), (5, 7, 13))
    draw = ImageDraw.Draw(canvas)
    try:
        ftitle = ImageFont.truetype("arialbd.ttf", 22)
        fsub = ImageFont.truetype("arial.ttf", 14)
        fcell = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        ftitle = fsub = fcell = ImageFont.load_default()

    draw.text((pad, 12), f"INT8 feature maps · layer {args.layer}", fill=(234, 242, 255), font=ftitle)
    draw.text((pad, 42),
              f"early-conv activations the accelerator streams · {C} channels @ {H}x{W}, "
              f"INT8 codes [-127,127] · showing first {nshow}",
              fill=(138, 155, 180), font=fsub)

    # turbo-ish colormap via matplotlib if available, else grayscale
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("inferno")
        use_cmap = True
    except Exception:
        use_cmap = False

    for idx in range(nshow):
        ch = codes[idx].astype(np.float32)
        lo, hi = ch.min(), ch.max()
        norm = (ch - lo) / (hi - lo + 1e-6)
        if use_cmap:
            rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        else:
            g = (norm * 255).astype(np.uint8)
            rgb = np.stack([g, g, g], axis=-1)
        tile = Image.fromarray(rgb, "RGB").resize((cell, cell), Image.NEAREST)
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell + pad)
        y0 = 70 + pad + r * (cell + label_h + pad)
        canvas.paste(tile, (x0, y0))
        draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline=(40, 52, 70), width=1)
        draw.text((x0 + 2, y0 + cell + 4), f"ch{idx}  [{int(lo)},{int(hi)}]",
                  fill=(150, 168, 195), font=fcell)

    out = out_dir / "sim_layer_activations.png"
    canvas.save(out)
    print(f"  layer={args.layer}  shape=({C},{H},{W})  act_scale={act_scale:.6g}")
    print(f"  montage -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
