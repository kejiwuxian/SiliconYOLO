#!/usr/bin/env python
"""Silicon YOLO — STEP 6: export per-layer INT8 constants for BRAM/ROM init.

hw_graph.json (REF-04) is the authority: for each Conv2D layer it declares the
exact `weight_mem_file` / `bias_mem_file` paths and the requant params. This
script reads the frozen INT8 weights (model/frozen/yolov10n_int8_frozen.pt) and
writes EXACTLY those files, plus matching `.coe`, per spec §4.3.3:

  * Format : ASCII hex, one value per line, MSB first
  * Width  : 8 bits / line (INT8 weights), 32 bits / line (INT32 bias)
  * Order  : [c_out][c_in][kh][kw] — output-channel major (row-major flatten)

Bias int32 = round(bias_fp32 / (weight_scale[oc] * act_scale_in)) so it adds
directly into the INT32 MAC accumulator before requantization.

Usage:
  python hwconst/export_consts.py      # run AFTER freeze.py and export_graph.py
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


def _hex(v: int, bits: int) -> str:
    """Two's-complement hex (MSB-first, fixed nibble width)."""
    return format(int(v) & ((1 << bits) - 1), "x").zfill(bits // 4)


def _write_mem(path: Path, values, bits: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_hex(v, bits) for v in values) + "\n")


def _write_coe(path: Path, values, bits: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(_hex(v, bits) for v in values)
    path.write_text("memory_initialization_radix=16;\n"
                    "memory_initialization_vector=\n" + body + ";\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=str(FROZEN))
    args = ap.parse_args()

    import torch

    frozen_path = Path(args.frozen)
    if not frozen_path.exists():
        print(f"!! frozen INT8 weights not found: {frozen_path} (run model/freeze.py)")
        return 1
    if not GRAPH.exists():
        print(f"!! hw_graph.json not found: {GRAPH} (run hwgraph/export_graph.py)")
        return 1

    blob = torch.load(str(frozen_path), map_location="cpu", weights_only=False)
    fl = blob["layers"]
    graph = json.loads(GRAPH.read_text())
    conv_layers = [L for L in graph["layers"] if L.get("op") == "Conv2D"]

    index = {}
    n_w = n_b = 0
    for L in conv_layers:
        name = L["name"]
        if name not in fl:
            print(f"  [warn] {name} not in frozen weights; skipping")
            continue
        entry = fl[name]
        w = entry["weight_int8"].numpy()           # [oc,ic,kh,kw] int8
        flat = w.reshape(-1)                        # row-major oc,ic,kh,kw  (spec §4.3.3)

        wmem = ROOT / L["weight_mem_file"]
        wcoe = wmem.with_suffix(".coe")
        _write_mem(wmem, flat, 8)
        _write_coe(wcoe, flat, 8)
        n_w += 1
        rec = {"name": name, "id": L["id"],
               "weight": {"file_mem": L["weight_mem_file"],
                          "file_coe": str(wcoe.relative_to(ROOT)).replace("\\", "/"),
                          "dtype": "int8", "bits": 8, "depth": int(flat.size),
                          "shape_oc_ic_kh_kw": list(w.shape),
                          "flatten_order": "c_out,c_in,kh,kw (row-major, MSB-first hex)"}}

        if L.get("bias_mem_file") and "bias_fp32" in entry:
            bias = entry["bias_fp32"].numpy()
            wscale = entry["weight_scale"].numpy()              # per-oc
            act_in = float(L["requant"]["act_scale_in"])
            denom = wscale * act_in
            denom[denom == 0] = 1e-12
            bias_i32 = [max(-(2**31), min(2**31 - 1, int(round(float(b) / float(d)))))
                        for b, d in zip(bias, denom)]
            bmem = ROOT / L["bias_mem_file"]
            bcoe = bmem.with_suffix(".coe")
            _write_mem(bmem, bias_i32, 32)
            _write_coe(bcoe, bias_i32, 32)
            n_b += 1
            rec["bias"] = {"file_mem": L["bias_mem_file"],
                           "file_coe": str(bcoe.relative_to(ROOT)).replace("\\", "/"),
                           "dtype": "int32", "bits": 32, "depth": len(bias_i32),
                           "formula": "round(bias_fp32 / (weight_scale[oc] * act_scale_in))",
                           "act_scale_in": round(act_in, 9)}
        index[name] = rec

    idx = {
        "task": "STEP 6 hwconst (.mem/.coe) export",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_frozen": str(frozen_path.relative_to(ROOT)).replace("\\", "/"),
        "authority": "hwgraph/hw_graph.json (filenames + requant)",
        "num_weight_files": n_w,
        "num_bias_files": n_b,
        "conventions": {
            "weight": "int8, 2's-complement hex, 2 nibbles/line, MSB-first, [c_out][c_in][kh][kw]",
            "bias": "int32, 2's-complement hex, 8 nibbles/line, MSB-first",
        },
        "layers": index,
    }
    (ROOT / "hwconst" / "index.json").write_text(json.dumps(idx, indent=2) + "\n")
    print("== STEP 6 hwconst export ==")
    print(f"  wrote {n_w} weight + {n_b} bias .mem/.coe pairs under hwconst/mem/")
    print(f"  index -> hwconst/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
