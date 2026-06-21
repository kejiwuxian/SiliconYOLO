#!/usr/bin/env python
"""Silicon YOLO — STEP 5: FREEZE the canonical hardware weights.

Because there is NO training, freezing is immediate. We freeze the **fused**
YOLOv10n (BN folded, NMS-free one2one head) — the exact float weights the INT8
quantization (hwconst/quant_scales.json) was computed against — plus the
quantized INT8 integer weights/biases per layer.

Outputs:
  model/frozen/yolov10n_fused_fp32.pt   — fused float state_dict (HW reference)
  model/frozen/yolov10n_int8_frozen.pt  — INT8 integer weights+biases+scales
  model/frozen/FROZEN.json              — manifest (hashes, shapes, provenance)

The INT8 .pt is the single source the hwconst exporter (.mem/.coe) reads.

Usage:
  python model/freeze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

FROZEN_DIR = ROOT / "model" / "frozen"
SCALES = ROOT / "hwconst" / "quant_scales.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--scales", default=str(SCALES))
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    from common import BASE_WEIGHTS, load_detection_model

    weights = Path(args.weights) if args.weights else BASE_WEIGHTS
    scales_path = Path(args.scales)
    if not scales_path.exists():
        print(f"!! quant scales not found: {scales_path} (run quant/ptq.py first)")
        return 1

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    print("== STEP 5 FREEZE ==")

    model = load_detection_model(weights)
    model.fuse()
    model.eval()

    # 1) fused FP32 reference
    fp32_path = FROZEN_DIR / "yolov10n_fused_fp32.pt"
    torch.save(model.state_dict(), str(fp32_path))
    print(f"  fused FP32 state_dict -> {fp32_path.name}")

    # 2) INT8 integer weights + biases per Conv2d (the HW payload)
    scales = json.loads(scales_path.read_text())["layers"]
    int8_layers = {}
    n_w = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if name not in scales:
            print(f"  [warn] no scale for {name}; skipping")
            continue
        w = m.weight.detach()
        oc = w.shape[0]
        ws = torch.tensor(scales[name]["weight"]["scales"]).view(oc, *([1] * (w.dim() - 1)))
        q = torch.clamp(torch.round(w / ws), -127, 127).to(torch.int8)
        entry = {
            "weight_int8": q,                                  # int8 tensor [oc,ic,kh,kw]
            "weight_scale": ws.flatten().to(torch.float32),    # per-channel
        }
        # Bias kept in higher precision: bias_int32 = round(bias / (w_scale*act_in_scale)).
        # act-in scale is layer-dependent; for a portable freeze we store FP32 bias
        # and the per-channel weight scale, and let the hwconst exporter combine
        # with the activation-input scale to produce the int32 bias.
        if m.bias is not None:
            entry["bias_fp32"] = m.bias.detach().to(torch.float32)
        int8_layers[name] = entry
        n_w += 1

    int8_path = FROZEN_DIR / "yolov10n_int8_frozen.pt"
    torch.save({
        "format": "silicon_yolo_int8_frozen_v1",
        "model": "yolov10n_fused_nmsfree",
        "scheme": "per_channel_symmetric_int8_weights",
        "layers": int8_layers,
    }, str(int8_path))
    print(f"  INT8 frozen weights ({n_w} layers) -> {int8_path.name}")

    manifest = {
        "task": "STEP 5 freeze",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "no_training": True,
        "source_base": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
        "quant_scales": str(scales_path.relative_to(ROOT)),
        "artifacts": {
            "fused_fp32": {"file": fp32_path.name, "sha256_16": _sha(fp32_path)},
            "int8_frozen": {"file": int8_path.name, "sha256_16": _sha(int8_path), "num_layers": n_w},
        },
        "stable_path": "model/frozen/yolov10n_int8_frozen.pt",
    }
    (FROZEN_DIR / "FROZEN.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  manifest -> FROZEN.json")
    print(f"  frozen path (stable): {manifest['stable_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
