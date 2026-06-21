#!/usr/bin/env python
"""Export a golden vector to plain hex/text files the SystemVerilog top-level TB
can $readmemh / $fscanf, so the RTL bench is driven by the SAME bit-exact data
the fixed-point model produced.

Writes (under rtl_tb/golden_hex/<id>/):
  input_rgb.hex      one 24-bit RGB pixel per line (hex), row-major HxW
                     (reconstructed from the preprocessed letterboxed frame)
  input_int8.hex     one INT8 (2's-comp hex byte) per line, CHW order
  detections.txt     golden detections: "x1 y1 x2 y2 score_u8 cls" per line
  meta.txt           H W C n_detections

Usage:
  python rtl_tb/export_golden_hex.py --id 0000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

VEC = ROOT / "golden" / "vectors"
OUT = ROOT / "rtl_tb" / "golden_hex"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="0000")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    import numpy as np

    vid = args.id
    vdir = VEC / vid
    if not vdir.exists():
        print(f"!! golden vector {vdir} not found (run golden/gen_golden_fixed.py)")
        return 1

    xin = np.load(vdir / "input_int8.npy")          # (1,3,H,W) int8
    dets = np.load(vdir / "detections_int.npz")
    boxes = dets["boxes"][0]                          # (300,4)
    score = dets["score_u8"][0].ravel()               # (300,)
    cls = dets["cls"][0].ravel()                       # (300,)

    odir = OUT / vid
    odir.mkdir(parents=True, exist_ok=True)

    _, C, H, W = xin.shape
    arr = xin[0]                                       # (C,H,W) int8

    # INT8 stream, CHW order, one 2's-complement hex byte per line
    with open(odir / "input_int8.hex", "w") as f:
        for c in range(C):
            for y in range(H):
                for x in range(W):
                    f.write(f"{int(arr[c, y, x]) & 0xFF:02x}\n")

    # 24-bit RGB per pixel (reconstruct uint8 from INT8 datapath input:
    # the preprocessor takes RGB; here we map INT8 code back to a uint8 pixel by
    # +128 offset purely so the TB has a representative video stream to frame).
    with open(odir / "input_rgb.hex", "w") as f:
        for y in range(H):
            for x in range(W):
                r = int(arr[0, y, x]) & 0xFF
                g = int(arr[1, y, x]) & 0xFF
                b = int(arr[2, y, x]) & 0xFF
                f.write(f"{(r << 16) | (g << 8) | b:06x}\n")

    # golden detections (only those above a nominal score, but dump all 300)
    n_keep = int((score > 0).sum())
    with open(odir / "detections.txt", "w") as f:
        for i in range(boxes.shape[0]):
            f.write(f"{int(boxes[i,0])} {int(boxes[i,1])} {int(boxes[i,2])} "
                    f"{int(boxes[i,3])} {int(score[i])} {int(cls[i])}\n")

    (odir / "meta.txt").write_text(f"{H} {W} {C} {n_keep}\n")
    print(f"  exported golden id={vid}: {C}x{H}x{W} INT8 + {boxes.shape[0]} det records")
    print(f"  -> {odir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
