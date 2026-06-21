#!/usr/bin/env python
"""Silicon YOLO — STEP B: bit-exact FIXED-POINT golden vectors for RTL.

The float goldens (now archived under golden/float_ref/) used FP32 intermediates.
This generator reimplements every Conv2d in INTEGER fixed-point EXACTLY as the
hardware requant unit will, so the dumped per-layer activations are what the
Cognichip RTL testbench checks bit-for-bit:

  For each Conv2d (params + scales from hwgraph/hw_graph.json requant):
    1. capture its real float input X (forward_pre_hook)
    2. quantize input to INT8:   Xq = clip(round(X / act_scale_in), -127, 127)
    3. INT8 weights Wq from model/frozen/yolov10n_int8_frozen.pt
    4. INT32 accumulate:         acc = conv2d(Xq, Wq)        # exact integer MAC
    5. add INT32 bias:           acc += round(bias / (w_scale[oc]*act_scale_in))
    6. requantize to INT8 out:   Yq = clip(round(acc * w_scale[oc]*act_scale_in
                                                  / act_scale_out), -127, 127)

  No floats in the datapath — steps 4-6 use float64 holding EXACT integers
  (|acc| << 2^31, exactly representable), so the result equals an int32 datapath.

Per layer we report max/mean abs error between the fixed-point dequantized output
(Yq * act_scale_out) and the captured FP32 reference conv output.

Final detections are dumped as integers: boxes as int pixel coords, score as
uint8 (round(score*255)), class id as int.

Outputs (per image id):
  golden/vectors/<id>/input_int8.npy         INT8 quantized input
  golden/vectors/<id>/layers_int8.npz         INT8 per-Conv2d output activations
  golden/vectors/<id>/acc_int32.npz           INT32 pre-requant accumulators
  golden/vectors/<id>/detections_int.npz       integer final detections
  golden/manifest_fixed.json                   contract + per-layer error report

Usage:
  python golden/gen_golden_fixed.py --num-images 4
  python golden/gen_golden_fixed.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

FROZEN = ROOT / "model" / "frozen" / "yolov10n_int8_frozen.pt"
GRAPH = ROOT / "hwgraph" / "hw_graph.json"
OUT_DEFAULT = ROOT / "golden" / "vectors"
MANIFEST = ROOT / "golden" / "manifest_fixed.json"
INPUT_ACT_SCALE = 1.0 / 127.0  # matches hw_graph input.act_scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from common import calib_images, load_detection_model

    if not FROZEN.exists() or not GRAPH.exists():
        print(f"!! need {FROZEN.name} (freeze.py) and {GRAPH.name} (export_graph.py)")
        return 1

    n = 2 if args.dry_run else args.num_images
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== STEP B fixed-point golden {'(DRY RUN)' if args.dry_run else ''} ==")

    # frozen INT8 weights + per-channel scales, keyed by layer name
    frozen = torch.load(str(FROZEN), map_location="cpu", weights_only=False)["layers"]
    graph = json.loads(GRAPH.read_text())
    req = {L["name"]: L for L in graph["layers"] if L.get("op") == "Conv2D"}

    model = load_detection_model().to("cpu")
    model.fuse()
    model.eval()
    name_of = {m: nm for nm, m in model.named_modules()}

    # accumulate per-layer error stats across images
    err_stats: dict[str, dict] = {}

    def make_pre_hook(name, store_q, store_acc, store_ref_out):
        rq = req[name]["requant"]
        act_in = float(rq["act_scale_in"])
        act_out = float(rq["act_scale_out"])
        wscale = np.asarray(rq["weight_scale_per_channel"], dtype=np.float64)
        fz = frozen[name]
        Wq = fz["weight_int8"].numpy().astype(np.float64)  # exact ints
        bias = fz.get("bias_fp32")
        bias = bias.numpy().astype(np.float64) if bias is not None else None

        def pre_hook(mod, inp):
            x = inp[0]
            xq = torch.clamp(torch.round(x / act_in), -127, 127)  # INT8 (as float)
            xq_np = xq.numpy().astype(np.float64)
            # exact integer conv via float64 (values are small integers)
            Wt = torch.from_numpy(Wq)
            acc = F.conv2d(torch.from_numpy(xq_np), Wt, bias=None,
                           stride=mod.stride, padding=mod.padding,
                           dilation=mod.dilation, groups=mod.groups)
            acc_np = acc.numpy()  # float64 holding exact int32 values
            oc = acc_np.shape[1]
            if bias is not None:
                denom = wscale * act_in
                denom[denom == 0] = 1e-12
                bias_i32 = np.round(bias / denom)
                acc_np = acc_np + bias_i32.reshape(1, oc, 1, 1)
            acc_np = np.rint(acc_np)  # ensure integer
            # requant to INT8 output
            mult = (wscale * act_in / act_out).reshape(1, oc, 1, 1)
            yq = np.clip(np.rint(acc_np * mult), -127, 127)
            store_q[name] = yq.astype(np.int8)
            store_acc[name] = acc_np.astype(np.int32)
            # for error: fixed dequant vs we record; ref filled by forward hook
            store_ref_out[name] = (yq * act_out)  # dequantized fixed output
        return pre_hook

    def make_fwd_hook(name, store_ref):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            if torch.is_tensor(t):
                store_ref[name] = t.detach().numpy().astype(np.float64)
        return hook

    layer_order = list(req.keys())
    records = []
    t0 = time.time()
    for i, (x, meta) in enumerate(calib_images(n, args.imgsz, args.dry_run)):
        store_q, store_acc, store_fix_deq, store_ref = {}, {}, {}, {}
        handles = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and nm in req:
                handles.append(m.register_forward_pre_hook(
                    make_pre_hook(nm, store_q, store_acc, store_fix_deq)))
                handles.append(m.register_forward_hook(make_fwd_hook(nm, store_ref)))

        xb = x.unsqueeze(0)
        # quantize the network input to INT8 (the S0 pre-processor output)
        xin_q = torch.clamp(torch.round(xb / INPUT_ACT_SCALE), -127, 127).numpy().astype(np.int8)
        with torch.no_grad():
            out = model(xb)
        for h in handles:
            h.remove()

        # per-layer float-vs-fixed error
        for nm in layer_order:
            if nm in store_fix_deq and nm in store_ref:
                d = np.abs(store_fix_deq[nm] - store_ref[nm])
                s = err_stats.setdefault(nm, {"max": 0.0, "sum": 0.0, "cnt": 0})
                s["max"] = max(s["max"], float(d.max()))
                s["sum"] += float(d.mean())
                s["cnt"] += 1

        # final detections as integers: [x1,y1,x2,y2,score_u8,class]
        det = out[0] if isinstance(out, (list, tuple)) else out
        det_np = det.detach().numpy()  # (1,300,6)
        boxes = np.rint(det_np[..., 0:4]).astype(np.int32)
        score_u8 = np.clip(np.rint(det_np[..., 4:5] * 255), 0, 255).astype(np.uint8)
        cls = np.rint(det_np[..., 5:6]).astype(np.int32)

        vid = f"{i:04d}"
        vdir = out_dir / vid
        vdir.mkdir(parents=True, exist_ok=True)
        np.save(vdir / "input_int8.npy", xin_q)
        np.savez_compressed(vdir / "layers_int8.npz", **store_q)
        np.savez_compressed(vdir / "acc_int32.npz", **store_acc)
        np.savez_compressed(vdir / "detections_int.npz",
                            boxes=boxes, score_u8=score_u8, cls=cls)
        records.append({"id": vid, "input_int8_shape": list(xin_q.shape),
                        "num_conv_layers": len(store_q),
                        "preprocess_meta": meta})
        print(f"  [{vid}] {len(store_q)} INT8 conv activations + {boxes.shape[1]} dets")

    err_report = {nm: {"max_abs_err": round(s["max"], 6),
                       "mean_abs_err": round(s["sum"] / max(s["cnt"], 1), 6)}
                  for nm, s in err_stats.items()}
    worst = sorted(err_report.items(), key=lambda kv: -kv[1]["max_abs_err"])[:5]

    manifest = {
        "task": "STEP B bit-exact fixed-point golden vectors",
        "model": "yolov10n (fused, NMS-free)",
        "precision": "INT8 datapath (INT8 weights, INT32 accumulate, INT8 requant)",
        "datapath": {
            "input_act_scale": round(INPUT_ACT_SCALE, 9),
            "weights": "per_channel_symmetric_int8",
            "activations": "per_tensor_symmetric_int8",
            "requant": "Yq = clip(round(acc_int32 * w_scale*act_in/act_out), -127, 127)",
        },
        "files_per_vector": {
            "input_int8.npy": "INT8 network input",
            "layers_int8.npz": "INT8 per-Conv2d output activations (RTL checks these)",
            "acc_int32.npz": "INT32 pre-requant accumulators",
            "detections_int.npz": "boxes:int32[x1y1x2y2], score_u8:uint8, cls:int32",
        },
        "detection_format": "[x1,y1,x2,y2] int pixels, score uint8 (round(score*255)), class int",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "imgsz": args.imgsz,
        "nms": False,
        "num_vectors": len(records),
        "layer_order": layer_order,
        "vectors": records,
        "float_vs_fixed_error": err_report,
        "worst5_layers_by_max_abs_err": [{"name": k, **v} for k, v in worst],
        "float_reference_dir": "golden/float_ref/",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote manifest -> {MANIFEST}  ({time.time()-t0:.1f}s)")
    print(f"  worst layer max-abs-err (float vs fixed):")
    for k, v in worst[:3]:
        print(f"    {k}: max={v['max_abs_err']}  mean={v['mean_abs_err']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
