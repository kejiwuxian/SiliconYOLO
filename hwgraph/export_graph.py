#!/usr/bin/env python
"""Silicon YOLO — STEP 6: lower the fused INT8 model to hwgraph/hw_graph.json.

`hw_graph.json` is the HW contract consumed by the Cognichip RTL track
(REF-04 in yolov10n_accel_spec.md). It is the SINGLE AUTHORITY for:
  * layer execution order + integer ids,
  * per-layer op / shapes / kernel / stride / padding / activation,
  * per-layer quant params + the integer **requant** (act_scale_in/out, shift_bits),
  * the .mem/.coe filenames each layer's weights/biases live in (so hwconst/
    export writes exactly these files).

Schema follows spec §4.3.1 (id, name, op, c_in, c_out, kernel, stride, padding,
activation, quant_bits_weights, quant_bits_act, weight_mem_file, bias_mem_file)
and adds detail fields (shapes, groups, depthwise, INT4 candidacy, requant).

The graph terminates at the NMS-free end-to-end detect head — no NMS node.

Side effect: enriches hwconst/quant_scales.json with a top-level `requant` map in
spec §4.3.2 shape (weight_scale, act_scale_in, act_scale_out, shift_bits) so
REF-05 is satisfied by a single file. Run order: ptq.py -> export_graph.py.

Usage:
  python hwgraph/export_graph.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

OUT = ROOT / "hwgraph" / "hw_graph.json"
SCALES = ROOT / "hwconst" / "quant_scales.json"

INPUT_ACT_SCALE = 1.0 / 127.0   # preprocessed input in [0,1], per-tensor symmetric
QUANT_SHIFT_MAX = 15            # spec P_QUANT_SHIFT_MAX


def _safe(name: str) -> str:
    return name.replace(".", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default=str(SCALES))
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    from common import calib_images, load_detection_model

    scales_path = Path(args.scales)
    quant_doc = json.loads(scales_path.read_text()) if scales_path.exists() else {"layers": {}}
    quant = quant_doc.get("layers", {})
    if not quant:
        print(f"  [warn] no quant scales at {scales_path}; requant params will be omitted")

    model = load_detection_model()
    model.fuse()
    model.eval()
    name_of = {m: n for n, m in model.named_modules()}

    CAPTURE = (nn.Conv2d, nn.SiLU, nn.MaxPool2d, nn.Upsample)
    trace = []
    order = {"i": 0}
    handles = []

    def mk_hook(module):
        def hook(mod, inp, out):
            t_in = inp[0] if isinstance(inp, (list, tuple)) and inp else inp
            t_out = out[0] if isinstance(out, (list, tuple)) else out
            rec = {"trace_idx": order["i"], "name": name_of.get(mod, "?"),
                   "op": type(mod).__name__}
            if torch.is_tensor(t_in):
                rec["in_shape"] = list(t_in.shape)
            if torch.is_tensor(t_out):
                rec["out_shape"] = list(t_out.shape)
            if isinstance(mod, nn.Conv2d):
                rec.update({"_in_channels": mod.in_channels, "_out_channels": mod.out_channels,
                            "_kernel": list(mod.kernel_size), "_stride": list(mod.stride),
                            "_padding": list(mod.padding), "_groups": mod.groups,
                            "_has_bias": mod.bias is not None})
            trace.append(rec)
            order["i"] += 1
        return hook

    for n, m in model.named_modules():
        if isinstance(m, CAPTURE):
            handles.append(m.register_forward_hook(mk_hook(m)))
    x, _ = next(calib_images(1, args.imgsz))
    with torch.no_grad():
        model(x.unsqueeze(0))
    for h in handles:
        h.remove()

    # Build the ordered layer contract. Conv2d layers get integer ids; structural
    # ops (pool/upsample/silu) are recorded inline for context.
    layers = []
    requant_map = {}
    conv_id = 0
    prev_act_out = INPUT_ACT_SCALE  # activation scale feeding the next conv
    for i, r in enumerate(trace):
        if r["op"] != "Conv2d":
            # keep structural ops as context entries (no id)
            layers.append({k: r[k] for k in ("op", "name", "in_shape", "out_shape") if k in r})
            continue
        name = r["name"]
        q = quant.get(name, {})
        wq = q.get("weight", {})
        aq = q.get("activation", {})
        # activation tag: a SiLU consuming this conv's output next
        nxt = trace[i + 1] if i + 1 < len(trace) else None
        act = "SiLU" if (nxt and nxt["op"] == "SiLU" and nxt.get("in_shape") == r.get("out_shape")) else "none"

        wscales = wq.get("scales", [])
        wscale_repr = (sum(wscales) / len(wscales)) if wscales else 0.0
        act_in = prev_act_out
        act_out = aq.get("scale", act_in) if aq.get("calibrated") else act_in
        # integer requant shift: out_int8 = round(acc_int32 * weight_scale*act_in/act_out)
        m = (wscale_repr * act_in / act_out) if act_out > 0 else 0.0
        shift_bits = int(min(QUANT_SHIFT_MAX, max(0, round(-math.log2(m))))) if m > 0 else QUANT_SHIFT_MAX
        int4 = bool(wq.get("int4_candidate", False))

        kh, kw = r["_kernel"]
        sh, sw = r["_stride"]
        ph, pw = r["_padding"]
        safe = _safe(name)
        layer = {
            "id": conv_id,
            "name": name,
            "op": "Conv2D",
            "c_in": r["_in_channels"],
            "c_out": r["_out_channels"],
            "kernel": kh if kh == kw else [kh, kw],
            "stride": sh if sh == sw else [sh, sw],
            "padding": ph if ph == pw else [ph, pw],
            "groups": r["_groups"],
            "depthwise": r["_groups"] == r["_in_channels"] and r["_groups"] > 1,
            "activation": act,
            # Frozen .mem weights are INT8 for ALL layers (the evaluated, frozen
            # precision). int4_candidate is an advisory flag for the RTL's
            # P_INT4_LAYER_MASK — an opportunistic re-quant for tolerant layers,
            # NOT applied to the frozen INT8 weights here.
            "quant_bits_weights": 8,
            "quant_bits_act": 8,
            "int4_candidate": int4,
            "in_shape": r.get("in_shape"),
            "out_shape": r.get("out_shape"),
            "weight_mem_file": f"hwconst/mem/layer_{conv_id:03d}_{safe}_w.mem",
            "bias_mem_file": (f"hwconst/mem/layer_{conv_id:03d}_{safe}_b.mem"
                              if r["_has_bias"] else None),
            "requant": {
                "weight_scale_per_channel": [round(s, 9) for s in wscales],
                "weight_scale": round(wscale_repr, 9),
                "act_scale_in": round(act_in, 9),
                "act_scale_out": round(act_out, 9),
                "shift_bits": shift_bits,
            },
        }
        layers.append(layer)
        requant_map[name] = {
            "weight_scale": round(wscale_repr, 9),
            "weight_scale_per_channel": [round(s, 9) for s in wscales],
            "act_scale_in": round(act_in, 9),
            "act_scale_out": round(act_out, 9),
            "shift_bits": shift_bits,
        }
        conv_id += 1
        prev_act_out = act_out

    graph = {
        "model": "yolov10n",
        "version": "1.0",
        "description": "Silicon YOLO HW contract — fused, NMS-free YOLOv10n (spec §4.3.1)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "imgsz": args.imgsz,
        "input": {"name": "images", "shape": [1, 3, args.imgsz, args.imgsz],
                  "layout": "NCHW", "dtype": "uint8->int8",
                  "preprocess": "letterbox, /255, pad=114", "act_scale": round(INPUT_ACT_SCALE, 9)},
        "nms": False,
        "note": "Graph ends at the end-to-end detect head; no NMS block in hardware.",
        "quant_scheme": {"weights": "per_channel_symmetric_int8",
                         "activations": "per_tensor_symmetric_int8",
                         "requant": "out = clip(round(acc_int32 * weight_scale*act_in/act_out))"},
        "num_conv": conv_id,
        "num_entries": len(layers),
        "layers": layers,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(graph, indent=2) + "\n")

    # Enrich quant_scales.json with spec §4.3.2 requant view (single-file REF-05).
    if quant:
        quant_doc["requant"] = requant_map
        quant_doc["requant_schema"] = "spec §4.3.2: weight_scale, act_scale_in, act_scale_out, shift_bits"
        scales_path.write_text(json.dumps(quant_doc, indent=2) + "\n")

    print("== STEP 6 hw_graph ==")
    print(f"  traced {len(trace)} ops; {conv_id} Conv2D layers in execution order")
    print(f"  NMS block: {graph['nms']}  (NMS-free head)")
    print(f"  wrote {OUT}")
    if quant:
        print(f"  enriched {scales_path.name} with requant map ({len(requant_map)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
