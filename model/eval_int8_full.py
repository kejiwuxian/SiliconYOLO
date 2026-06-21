#!/usr/bin/env python
"""Silicon YOLO — STEP A: full weight + activation INT8 fake-quant eval.

The weight-only eval (model/eval_int8.json, mAP50-95 0.3762) quantized only the
Conv2d weights. This closes the loop by ALSO fake-quantizing every Conv2d output
activation with the per-tensor symmetric scale calibrated in
hwconst/quant_scales.json — i.e. the full INT8 datapath the hardware runs.

Mechanism:
  * weights: per-output-channel symmetric INT8, W_q = round(W/s_oc)*s_oc
  * activations: per-tensor symmetric INT8 forward hook on each Conv2d output,
    A_q = clamp(round(A/s), -127, 127) * s   (s from the calibrated activation scale)

Optionally simulates INT4 weights on the flagged tolerant layers (--int4) to show
the accuracy held / recovered.

Usage:
  python model/eval_int8_full.py --tag int8_full
  python model/eval_int8_full.py --tag int8_full_int4 --int4   # INT4 tolerant layers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

SCALES = ROOT / "hwconst" / "quant_scales.json"


def _fake_quant_weights(model, layers: dict, use_int4: bool, int4_skip_head: bool = False) -> tuple[int, int]:
    """Per-channel symmetric fake-quant of every Conv2d weight, in place.
    Returns (num_int8_layers, num_int4_layers)."""
    import torch
    import torch.nn as nn

    n8 = n4 = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d) or name not in layers:
            continue
        wq = layers[name]["weight"]
        sc = wq["scales"]
        is_int4 = use_int4 and wq.get("int4_candidate", False)
        # the detect head (model.23.*) is accuracy-critical; keep it INT8.
        if int4_skip_head and name.startswith("model.23"):
            is_int4 = False
        qmax = 7 if is_int4 else 127
        w = m.weight.detach()
        oc = w.shape[0]
        if is_int4:
            # re-derive the int4 per-channel scale from the channel max (qmax=7)
            amax = w.abs().reshape(oc, -1).amax(dim=1)
            scale = (amax / 7).clamp_min(1e-12).view(oc, *([1] * (w.dim() - 1)))
        else:
            scale = torch.tensor(sc, dtype=w.dtype).view(oc, *([1] * (w.dim() - 1)))
        q = torch.clamp(torch.round(w / scale), -qmax, qmax)
        m.weight.data.copy_(q * scale)
        n4 += int(is_int4)
        n8 += int(not is_int4)
    return n8, n4


def _attach_act_quant(model, layers: dict):
    """Register forward hooks that fake-quant each Conv2d OUTPUT (per-tensor INT8)."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--scales", default=str(SCALES))
    ap.add_argument("--tag", default="int8_full")
    ap.add_argument("--int4", action="store_true",
                    help="simulate INT4 weights on flagged tolerant layers")
    ap.add_argument("--int4-skip-head", action="store_true",
                    help="keep the detect head (model.23.*) at INT8 even under --int4")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    from common import BASE_WEIGHTS, COCO_ROOT

    weights = Path(args.weights) if args.weights else BASE_WEIGHTS
    scales_path = Path(args.scales)
    if not scales_path.exists():
        print(f"!! scales not found: {scales_path}")
        return 1
    layers = json.loads(scales_path.read_text())["layers"]

    print(f"== STEP A full W+A INT8 eval [{args.tag}] ==")
    print(f"  weights={weights}  int4={args.int4}  device={args.device}")

    t0 = time.time()
    yolo = YOLO(str(weights))
    yolo.model.fuse()
    n8, n4 = _fake_quant_weights(yolo.model, layers, args.int4, args.int4_skip_head)
    print(f"  weight fake-quant: {n8} INT8 + {n4} INT4 layers")
    handles = _attach_act_quant(yolo.model, layers)
    print(f"  activation fake-quant hooks: {len(handles)} Conv2d outputs")

    data_yaml = COCO_ROOT / "coco-val.yaml"
    metrics = yolo.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device,
                       verbose=False, save_json=True, plots=False)
    for h in handles:
        h.remove()

    map5095, map50 = float(metrics.box.map), float(metrics.box.map50)
    out = {
        "tag": args.tag,
        "weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
        "precision": "int8_weight+activation_fakequant" + ("_int4tolerant" if args.int4 else ""),
        "weight_int8_layers": n8,
        "weight_int4_layers": n4,
        "activation_quant_hooks": len(handles),
        "imgsz": args.imgsz,
        "metric": "pycocotools (ultralytics validator)",
        "mAP50_95": round(map5095, 5),
        "mAP50": round(map50, 5),
        "reference": {"fp32_mAP50_95": 0.37935, "weight_only_int8_mAP50_95": 0.37623},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    suffix = ("_int4_skiphead" if args.int4_skip_head else "_int4") if args.int4 else ""
    out_path = ROOT / "model" / f"eval_{args.tag}{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  W+A INT8: mAP50-95={map5095:.4f}  mAP50={map50:.4f}")
    print(f"  vs FP32 0.3794 / weight-only 0.3762")
    print(f"  wrote {out_path}  ({out['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
