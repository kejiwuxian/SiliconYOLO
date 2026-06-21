#!/usr/bin/env python
"""Silicon YOLO — shared helpers (paths, preprocessing, model loading, COCO).

Single source of truth for the data plumbing every stage reuses:
  * locating the pretrained base checkpoint + the (reused) genesys2 COCO val set,
  * letterbox preprocessing matched to ultralytics (640, pad=114, scaleup),
  * loading the YOLOv10n nn.Module for hook-based calibration / capture,
  * a deterministic calibration-image iterator.

No training anywhere — this project only *uses* an off-the-shelf pretrained model.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Pretrained base ---------------------------------------------------------
BASE_WEIGHTS = ROOT / "model" / "yolov10n.pt"
BASE_NAME = "yolov10n"

# --- Reused COCO val2017 from the genesys2 project (do NOT re-download) -------
COCO_ROOT = Path(r"D:\Projects\FPGA\genesys2\datasets\coco")
COCO_VAL_IMAGES = COCO_ROOT / "images" / "val2017"
COCO_VAL_ANN = COCO_ROOT / "annotations" / "instances_val2017.json"

IMGSZ = 640
PAD_VALUE = 114


def coco_val_image_paths(n: int | None = None) -> list[Path]:
    """Sorted list of COCO val2017 .jpg paths (first n if given)."""
    paths = sorted(COCO_VAL_IMAGES.glob("*.jpg"))
    return paths[:n] if n else paths


def letterbox(img, size: int = IMGSZ):
    """Resize-with-padding to (size,size); return (CHW float32 [0,1], meta).

    meta = dict(scale, pad_x, pad_y, orig_w, orig_h) — needed to map detections
    back to original-image coordinates for COCO eval.
    """
    import numpy as np
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img_r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img_r, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype="float32") / 255.0  # HWC
    chw = arr.transpose(2, 0, 1)  # CHW
    meta = {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "orig_w": w, "orig_h": h}
    return chw, meta


def load_detection_model(weights: Path = BASE_WEIGHTS):
    """Load the raw YOLOv10n nn.Module (eval, no grad) for hooks/capture."""
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    model = yolo.model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def torch_device(spec: str):
    """Normalize an ultralytics-style device spec ('0', 'cpu', 'cuda:0') to a
    torch device. Bare integer strings mean CUDA ordinals."""
    import torch

    if spec is None or spec == "":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    s = str(spec)
    if s.isdigit():
        return torch.device(f"cuda:{s}")
    return torch.device(s)


def calib_images(n: int, imgsz: int = IMGSZ, dry_run: bool = False):
    """Yield up to n (CHW float32 [0,1] tensor, meta) pairs from COCO val
    (synthetic fallback if images are absent)."""
    import torch

    paths = coco_val_image_paths(n)
    if not paths:
        if not dry_run:
            print("  [warn] no COCO val images found — using synthetic calibration data")
        for i in range(n):
            g = torch.Generator().manual_seed(1000 + i)
            yield torch.rand(3, imgsz, imgsz, generator=g), None
        return
    for p in paths:
        chw, meta = letterbox(p, imgsz)
        yield torch.from_numpy(chw), meta


if __name__ == "__main__":
    print("ROOT          ", ROOT)
    print("BASE_WEIGHTS  ", BASE_WEIGHTS, "exists:", BASE_WEIGHTS.exists())
    print("COCO_VAL_IMG  ", COCO_VAL_IMAGES, "exists:", COCO_VAL_IMAGES.exists())
    print("COCO_VAL_ANN  ", COCO_VAL_ANN, "exists:", COCO_VAL_ANN.exists())
    print("num val imgs  ", len(coco_val_image_paths()))
