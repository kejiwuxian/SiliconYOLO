#!/usr/bin/env python
"""Silicon YOLO — STEP 6: golden test-vector generator (RTL verification).

For a small fixed COCO image set, dumps the tensors an RTL bench needs to check
the fixed-weight datapath bit-for-bit:

  golden/vectors/<id>/input.npy    preprocessed NCHW input (fp32 [0,1])
  golden/vectors/<id>/layers.npz   every Conv2d / SiLU / MaxPool / Upsample output
  golden/vectors/<id>/output.npz   final NMS-free detections

plus golden/manifest.json (layer order, shapes, dtypes, provenance) — the
contract rtl_tb reads (REF-08). Captured on the FUSED model (BN folded, NMS-free
one2one head) so layer names match hw_graph.json / hwconst/.

Precision: FP32 reference + the exact file/format contract. The bit-accurate
fixed-point golden (driven by hwconst/quant_scales.json requant) is the
production follow-up; this gives RTL bring-up the reference tensors + format now.

Usage:
  python golden/gen_golden.py --num-images 4
  python golden/gen_golden.py --dry-run        # 2 imgs, CPU, code-path check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

OUT_DEFAULT = ROOT / "golden" / "vectors"
MANIFEST = ROOT / "golden" / "manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    from common import BASE_WEIGHTS, calib_images, load_detection_model, torch_device

    n = 2 if args.dry_run else args.num_images
    device = torch_device("cpu" if args.dry_run else args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== STEP 6 golden gen {'(DRY RUN)' if args.dry_run else ''} ==")
    print(f"  base={BASE_WEIGHTS.name}  images={n}  device={device}  imgsz={args.imgsz}")

    model = load_detection_model().to(device)
    model.fuse()
    model.eval()
    name_of = {m: nm for nm, m in model.named_modules()}
    capture_types = (nn.Conv2d, nn.SiLU, nn.MaxPool2d, nn.Upsample)

    def attach():
        store, handles = {}, []
        for nm, m in model.named_modules():
            if isinstance(m, capture_types):
                def _mk(key):
                    def _hook(mod, inp, out):
                        t = out[0] if isinstance(out, (list, tuple)) else out
                        if torch.is_tensor(t):
                            store[key] = t.detach().cpu().numpy()
                    return _hook
                handles.append(m.register_forward_hook(_mk(nm)))
        return store, handles

    layer_order = None
    records = []
    t0 = time.time()
    for i, (x, meta) in enumerate(calib_images(n, args.imgsz, args.dry_run)):
        store, handles = attach()
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(xb)
        for h in handles:
            h.remove()

        vid = f"{i:04d}"
        vdir = out_dir / vid
        vdir.mkdir(parents=True, exist_ok=True)
        np.save(vdir / "input.npy", xb.cpu().numpy())
        np.savez_compressed(vdir / "layers.npz", **store)

        flat = out if isinstance(out, (list, tuple)) else [out]
        out_arrs = {f"out{j}": t.detach().cpu().numpy()
                    for j, t in enumerate(flat) if torch.is_tensor(t)}
        np.savez_compressed(vdir / "output.npz", **out_arrs)

        if layer_order is None:
            layer_order = list(store.keys())
        records.append({"id": vid, "input_shape": list(xb.shape),
                        "num_layers": len(store),
                        "output_keys": {k: list(v.shape) for k, v in out_arrs.items()},
                        "preprocess_meta": meta})
        print(f"  [{vid}] captured {len(store)} layer tensors -> {vdir}")

    manifest = {
        "task": "STEP 6 golden vectors",
        "model": "yolov10n (fused, NMS-free)",
        "precision": "fp32 reference",
        "note": ("FP32 reference tensors + file/format contract. Bit-accurate "
                 "fixed-point golden (per hwconst/quant_scales.json requant) is "
                 "the production follow-up."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "imgsz": args.imgsz,
        "nms": False,
        "num_vectors": len(records),
        "layer_order": layer_order,
        "vectors": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote manifest -> {MANIFEST}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
