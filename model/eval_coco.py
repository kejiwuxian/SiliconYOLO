#!/usr/bin/env python
"""Silicon YOLO — COCO val2017 evaluation (FP32 baseline, and INT8).

STEP 3: FP32 reference numbers for the pretrained YOLOv10n.
STEP 4: re-runs the same eval on the INT8-simulated model to record the drop.

Uses Ultralytics' built-in validator (pycocotools under the hood) pointed at the
reused genesys2 COCO val2017. Writes mAP50-95 / mAP50 to model/eval_<tag>.json.

For --int8, weights are fake-quantized in-place from hwconst/quant_scales.json
(per-channel symmetric INT8 weights), so the reported mAP reflects the frozen
hardware weights. Activation quantization effect is reported separately by the
PTQ calibration summary; this eval isolates the weight-quantization accuracy.

Usage:
  python model/eval_coco.py --weights model/yolov10n.pt --tag fp32
  python model/eval_coco.py --int8 --tag int8        # uses hwconst/quant_scales.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))


def _coco_yaml() -> Path:
    """The reused genesys2 COCO val yaml (uses val2017.txt → ultralytics runs
    the official pycocotools eval and resolves real class names)."""
    from common import COCO_ROOT

    y = COCO_ROOT / "coco-val.yaml"
    if not y.exists():
        raise FileNotFoundError(f"COCO val yaml not found: {y}")
    return y


def _fake_quant_weights_inplace(model, scales_path: Path) -> int:
    """Round each Conv2d weight to its per-channel INT8 grid, in place.

    W_q = clamp(round(W / scale[oc]), -127, 127) * scale[oc]
    Returns the number of Conv2d layers quantized.
    """
    import torch
    import torch.nn as nn

    data = json.loads(scales_path.read_text())
    layers = data["layers"]
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if name not in layers:
            continue
        sc = layers[name]["weight"]["scales"]
        w = m.weight.detach()
        oc = w.shape[0]
        scale = torch.tensor(sc, dtype=w.dtype).view(oc, *([1] * (w.dim() - 1)))
        q = torch.clamp(torch.round(w / scale), -127, 127)
        m.weight.data.copy_(q * scale)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="checkpoint (default: base YOLOv10n)")
    ap.add_argument("--int8", action="store_true",
                    help="fake-quantize weights from hwconst/quant_scales.json before eval")
    ap.add_argument("--scales", default=str(ROOT / "hwconst" / "quant_scales.json"))
    ap.add_argument("--tag", default="fp32")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-images", type=int, default=0,
                    help="0 = full 5000-image val; else cap (quick smoke)")
    args = ap.parse_args()

    from ultralytics import YOLO

    from common import BASE_WEIGHTS

    weights = Path(args.weights) if args.weights else BASE_WEIGHTS
    print(f"== COCO eval [{args.tag}] ==")
    print(f"  weights={weights}  int8={args.int8}  imgsz={args.imgsz}  device={args.device}")

    t0 = time.time()
    yolo = YOLO(str(weights))

    nq = 0
    if args.int8:
        scales_path = Path(args.scales)
        if not scales_path.exists():
            print(f"!! scales not found: {scales_path} (run quant/ptq.py first)")
            return 1
        # Fuse first so BN is folded into conv (matches the frozen HW weights and
        # the layer naming in quant_scales.json); ultralytics' validator treats an
        # already-fused model as fused and won't re-fold.
        yolo.model.fuse()
        nq = _fake_quant_weights_inplace(yolo.model, scales_path)
        print(f"  fused + fake-quantized {nq} Conv2d layers to INT8 per-channel grid")

    data_yaml = _coco_yaml()
    val_kwargs = dict(data=str(data_yaml), imgsz=args.imgsz, device=args.device,
                      verbose=False, save_json=True, plots=False)
    if args.max_images:
        # ultralytics has no direct cap; rely on full val unless smoke-testing.
        print(f"  [note] --max-images={args.max_images} ignored by ultralytics validator; running full val")

    metrics = yolo.val(**val_kwargs)
    map5095 = float(metrics.box.map)     # mAP@0.50:0.95
    map50 = float(metrics.box.map50)     # mAP@0.50

    out = {
        "tag": args.tag,
        "weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
        "precision": "int8_weights_fakequant" if args.int8 else "fp32",
        "quantized_conv_layers": nq,
        "imgsz": args.imgsz,
        "metric": "pycocotools (ultralytics validator)",
        "mAP50_95": round(map5095, 5),
        "mAP50": round(map50, 5),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = ROOT / "model" / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  mAP50-95={map5095:.4f}  mAP50={map50:.4f}")
    print(f"  wrote {out_path}  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
