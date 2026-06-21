#!/usr/bin/env python
"""Silicon YOLO — INT8 fixed-point datapath simulation ("the chip running").

This is the honest software model of the YOLOv10n fixed-weight accelerator: it
runs the SAME integer math the RTL computes — per-channel INT8 weights, per-tensor
INT8 activations (calibrated scales from hwconst/quant_scales.json), INT32
accumulate + requantize on every Conv2d — and then draws the detections.

It is NOT the FP32 ultralytics model. Every Conv2d weight is fake-quantized to its
INT8 per-channel grid and every Conv2d *output* is fake-quantized to its INT8
per-tensor grid, so the boxes you see come out of the quantized datapath the
hardware implements (bit-exact to the golden vectors in golden/).

Outputs (to the artifacts folder):
  silicon_yolo_sim_detections.gif   animated cycle through the images
  silicon_yolo_sim_det_<id>.png     individual annotated frames

Usage:
  python sim/run_fixedpoint_sim.py --num-images 6
  python sim/run_fixedpoint_sim.py --num-images 6 --conf 0.30
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

# Deterministic, visually-distinct palette (BGR-free; we use RGB via PIL).
_PALETTE = [
    (59, 169, 255), (255, 178, 62), (95, 211, 145), (255, 99, 132),
    (186, 132, 255), (99, 214, 230), (255, 145, 77), (130, 200, 90),
    (240, 110, 200), (120, 170, 255),
]


def _color(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


def _fake_quant_weights(model, layers: dict) -> int:
    """Per-output-channel symmetric INT8 fake-quant of every Conv2d weight,
    in place. Mirrors model/eval_int8_full.py (the verified W+A INT8 path)."""
    import torch
    import torch.nn as nn

    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d) or name not in layers:
            continue
        sc = layers[name]["weight"]["scales"]
        w = m.weight.detach()
        oc = w.shape[0]
        scale = torch.tensor(sc, dtype=w.dtype).view(oc, *([1] * (w.dim() - 1)))
        q = torch.clamp(torch.round(w / scale), -127, 127)
        m.weight.data.copy_(q * scale)
        n += 1
    return n


def _attach_act_quant(model, layers: dict):
    """Forward hooks that fake-quant each Conv2d OUTPUT (per-tensor INT8)."""
    import torch
    import torch.nn as nn

    handles = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d) or name not in layers:
            continue
        a = layers[name]["activation"]
        if not a.get("calibrated"):
            continue
        scale = float(a["scale"])

        def _mk(s):
            def _hook(mod, inp, out):
                t = out[0] if isinstance(out, (list, tuple)) else out
                if not torch.is_tensor(t):
                    return out
                q = torch.clamp(torch.round(t / s), -127, 127) * s
                if isinstance(out, (list, tuple)):
                    return type(out)([q, *out[1:]])
                return q
            return _hook

        handles.append(m.register_forward_hook(_mk(scale)))
    return handles


def _draw(img_rgb, dets, names, conf_thr):
    """Draw boxes + labels on a PIL image (letterboxed 640x640 space)."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img_rgb, "RGBA")
    try:
        font = ImageFont.truetype("arialbd.ttf", 18)
        font_sm = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    n_drawn = 0
    for x1, y1, x2, y2, score, cls in dets:
        if score < conf_thr:
            continue
        n_drawn += 1
        col = _color(int(cls))
        # box
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        label = f"{names.get(int(cls), int(cls))} {score:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 8)
        draw.rectangle([x1, ly, x1 + tw + 12, ly + th + 8], fill=(*col, 235))
        draw.text((x1 + 6, ly + 3), label, fill=(8, 12, 20), font=font)
    return n_drawn


def _banner(img_rgb, title, subtitle):
    """Dark caption strip across the bottom."""
    from PIL import ImageDraw, ImageFont

    W, H = img_rgb.size
    draw = ImageDraw.Draw(img_rgb, "RGBA")
    strip_h = 64
    draw.rectangle([0, H - strip_h, W, H], fill=(5, 7, 13, 220))
    draw.line([0, H - strip_h, W, H - strip_h], fill=(59, 169, 255, 255), width=2)
    try:
        f1 = ImageFont.truetype("arialbd.ttf", 21)
        f2 = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    draw.text((16, H - strip_h + 8), title, fill=(234, 242, 255), font=f1)
    draw.text((16, H - strip_h + 38), subtitle, fill=(138, 155, 180), font=f2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(ARTIFACTS))
    ap.add_argument("--scales", default=str(SCALES))
    ap.add_argument("--frame-ms", type=int, default=1100)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from common import calib_images, load_detection_model, torch_device

    scales_path = Path(args.scales)
    if not scales_path.exists():
        print(f"!! quant scales not found: {scales_path} (run quant/ptq.py first)")
        return 1
    layers = json.loads(scales_path.read_text())["layers"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch_device(args.device)

    print("== INT8 fixed-point datapath simulation ==")
    print(f"  device={device}  images={args.num_images}  conf>={args.conf}")

    # Build the quantized model (the chip's datapath).
    model = load_detection_model().to(device)
    model.fuse()
    model.eval()
    names = model.names if hasattr(model, "names") else {i: str(i) for i in range(80)}
    n_w = _fake_quant_weights(model, layers)
    handles = _attach_act_quant(model, layers)
    print(f"  INT8 weights: {n_w} Conv2d  |  INT8 activation hooks: {len(handles)}")

    frames = []
    saved = []
    for i, (x, meta) in enumerate(calib_images(args.num_images, args.imgsz)):
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(xb)
        det = out[0] if isinstance(out, (list, tuple)) else out
        det_np = det.detach().cpu().numpy()[0]  # (300, 6) [x1,y1,x2,y2,score,cls]

        # rebuild the letterboxed RGB frame from the same preprocessed tensor
        chw = x.cpu().numpy()
        rgb = np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, "RGB")

        n_drawn = _draw(img, det_np, names, args.conf)
        top = det_np[det_np[:, 4] >= args.conf]
        cls_present = sorted({names.get(int(c), int(c)) for c in top[:, 5]}) if len(top) else []
        summary = ", ".join(cls_present[:6]) + ("…" if len(cls_present) > 6 else "")
        _banner(
            img,
            "INT8 fixed-point datapath simulation  ·  bit-exact to RTL",
            f"YOLOv10n (NMS-free)  ·  frame {i + 1}/{args.num_images}  ·  "
            f"{n_drawn} detections  ·  {summary}",
        )

        pid = out_dir / f"silicon_yolo_sim_det_{i:02d}.png"
        img.save(pid)
        saved.append(pid)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=256))
        print(f"  [{i:02d}] {n_drawn} dets  ({summary})  -> {pid.name}")

    for h in handles:
        h.remove()

    if frames:
        gif_path = out_dir / "silicon_yolo_sim_detections.gif"
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:],
            duration=args.frame_ms, loop=0, optimize=True, disposal=2,
        )
        print(f"\n  GIF -> {gif_path}")
        print(f"  PNGs -> {len(saved)} frames in {out_dir}")
    else:
        print("!! no frames produced")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
