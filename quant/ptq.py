#!/usr/bin/env python
"""Silicon YOLO — STEP 4: INT8 post-training quantization (PTQ) + calibration.

Operates on the **fused** YOLOv10n (BN folded into conv, one2one NMS-free head
only) — the canonical hardware model. Produces hwconst/quant_scales.json:

  * per-output-channel symmetric INT8 weight scales  scale[oc] = max|W[oc]| / 127
  * per-tensor symmetric INT8 activation scales from a calibration sweep over a
    COCO val subset (running min/max at each Conv2d output)
  * per-layer INT4 candidacy flag (low-dynamic-range layers tolerant of 4 bits)

No training. Static PTQ only.

Usage:
  python quant/ptq.py --calib-images 256
  python quant/ptq.py --dry-run            # 4 imgs, CPU, code-path check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

OUT_DEFAULT = ROOT / "hwconst" / "quant_scales.json"
INT8_QMAX = 127
# A layer is an INT4 candidate when its quantized weights barely use the INT8
# range — i.e. the per-channel max |q| is small, so 4 bits (qmax 7) still covers
# the distribution with negligible extra error. Heuristic threshold below.
INT4_Q_USAGE_THRESH = 7  # if max|round(W/scale_int4)| fits in int4 for >=90% channels


def _fuse(model):
    """Fold BN into conv (and drop YOLOv10's one2many head)."""
    model.fuse()
    return model


def weight_scales_per_channel(conv) -> dict:
    import torch

    w = conv.weight.detach()
    oc = w.shape[0]
    amax = w.abs().reshape(oc, -1).amax(dim=1)
    scales = (amax / INT8_QMAX).clamp_min(1e-12)
    # INT4 candidacy: re-quantize with int4 grid using the SAME per-channel range
    # and check how much error 4 bits would add relative to 8 bits.
    s8 = scales.view(oc, *([1] * (w.dim() - 1)))
    q8 = torch.clamp(torch.round(w / s8), -127, 127)
    deq8 = q8 * s8
    s4 = (amax / 7).clamp_min(1e-12).view(oc, *([1] * (w.dim() - 1)))
    q4 = torch.clamp(torch.round(w / s4), -7, 7)
    deq4 = q4 * s4
    # relative extra MSE from going 8->4 bits
    denom = (w.pow(2).mean()).item() + 1e-12
    err8 = (w - deq8).pow(2).mean().item() / denom
    err4 = (w - deq4).pow(2).mean().item() / denom
    int4_ok = err4 < 0.03  # <3% relative weight MSE at int4 => tolerant layer
    return {
        "scheme": "per_channel_symmetric",
        "bits": 8,
        "num_channels": int(oc),
        "scales": [round(float(s), 9) for s in scales],
        "rel_mse_int8": round(err8, 6),
        "rel_mse_int4": round(err4, 6),
        "int4_candidate": bool(int4_ok),
    }


class _RangeCollector:
    def __init__(self):
        self.min = None
        self.max = None
        self.count = 0

    def __call__(self, module, inp, out):
        import torch

        t = out[0] if isinstance(out, (list, tuple)) else out
        if not torch.is_tensor(t):
            return
        lo, hi = float(t.min()), float(t.max())
        self.min = lo if self.min is None else min(self.min, lo)
        self.max = hi if self.max is None else max(self.max, hi)
        self.count += 1


def activation_scale(c: "_RangeCollector") -> dict:
    if c.min is None:
        return {"scheme": "per_tensor_symmetric", "bits": 8, "calibrated": False}
    amax = max(abs(c.min), abs(c.max))
    scale = max(amax / INT8_QMAX, 1e-12)
    return {
        "scheme": "per_tensor_symmetric",
        "bits": 8,
        "calibrated": True,
        "observed_min": round(c.min, 6),
        "observed_max": round(c.max, 6),
        "scale": round(scale, 9),
        "zero_point": 0,
        "num_batches": c.count,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="checkpoint (default base YOLOv10n)")
    ap.add_argument("--calib-images", type=int, default=256)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    from common import BASE_WEIGHTS, calib_images, load_detection_model, torch_device

    n_calib = 4 if args.dry_run else args.calib_images
    device = torch_device("cpu" if args.dry_run else args.device)
    weights = Path(args.weights) if args.weights else BASE_WEIGHTS

    print(f"== STEP 4 PTQ {'(DRY RUN)' if args.dry_run else ''} ==")
    print(f"  weights={weights}  calib_images={n_calib}  device={device}  imgsz={args.imgsz}")

    t0 = time.time()
    model = _fuse(load_detection_model(weights)).to(device)
    convs = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    print(f"  fused model: {sum(p.numel() for p in model.parameters())/1e6:.3f} M params, "
          f"{len(convs)} Conv2d layers")

    # 1) per-channel weight scales (+ INT4 candidacy)
    layers = {}
    for name, conv in convs:
        layers[name] = {
            "type": "Conv2d",
            "out_channels": int(conv.weight.shape[0]),
            "in_channels": int(conv.weight.shape[1]),
            "kernel": list(conv.weight.shape[2:]),
            "stride": list(conv.stride),
            "padding": list(conv.padding),
            "groups": int(conv.groups),
            "has_bias": conv.bias is not None,
            "weight": weight_scales_per_channel(conv),
        }

    # 2) activation calibration
    collectors = {name: _RangeCollector() for name, _ in convs}
    handles = [conv.register_forward_hook(collectors[name]) for name, conv in convs]
    n_done = 0
    with torch.no_grad():
        for x, _meta in calib_images(n_calib, args.imgsz, args.dry_run):
            model(x.unsqueeze(0).to(device))
            n_done += 1
            if not args.dry_run and n_done % 50 == 0:
                print(f"    calibrated {n_done}/{n_calib}")
    for h in handles:
        h.remove()
    print(f"  calibration: ran {n_done} image(s) through {len(convs)} hooked layers")

    for name in layers:
        layers[name]["activation"] = activation_scale(collectors[name])

    int4 = [n for n, l in layers.items() if l["weight"]["int4_candidate"]]
    payload = {
        "task": "STEP 4 INT8 PTQ scales",
        "stage": "dry-run scaffold" if args.dry_run else "calibrated",
        "model": "yolov10n (fused, NMS-free)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
        "scheme": {
            "weights": "per_channel_symmetric_int8",
            "activations": "per_tensor_symmetric_int8",
            "qmax_int8": INT8_QMAX,
            "qmax_int4": 7,
        },
        "num_calib_images": n_done,
        "num_conv_layers": len(convs),
        "int4_candidate_layers": int4,
        "num_int4_candidates": len(int4),
        "layers": layers,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        out = out.with_name(out.stem + "_dryrun" + out.suffix)
        payload["_warning"] = "DRY RUN — not real calibration; do not use for hardware."
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {len(layers)} layer scale sets -> {out}")
    print(f"  INT4 candidates: {len(int4)}/{len(convs)} layers")
    print(f"  done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
