#!/usr/bin/env python
"""Silicon YOLO — fetch the official pretrained COCO base checkpoint.

NO TRAINING. Downloads the upstream Ultralytics YOLOv10n COCO checkpoint (if not
already cached) into model/yolov10n.pt and prints a structural summary used by
docs/MODEL_CHOICE.md.

Usage:
  python model/fetch_base.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))


def main() -> int:
    import torch.nn as nn
    from ultralytics import YOLO

    from common import BASE_WEIGHTS

    # YOLO() downloads from github.com/ultralytics/assets if the file is absent.
    yolo = YOLO("yolov10n.pt" if not BASE_WEIGHTS.exists() else str(BASE_WEIGHTS))
    # Make sure the canonical copy lives at model/yolov10n.pt.
    if not BASE_WEIGHTS.exists():
        import shutil
        shutil.move("yolov10n.pt", str(BASE_WEIGHTS))

    model = yolo.model
    n_params = sum(p.numel() for p in model.parameters())
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    print(f"base checkpoint : {BASE_WEIGHTS}")
    print(f"task            : {yolo.task}")
    print(f"params          : {n_params/1e6:.3f} M")
    print(f"conv2d layers   : {n_conv}")
    print(f"end2end (NMS-free): {getattr(model, 'end2end', None)}")
    print(f"num classes     : {model.nc if hasattr(model, 'nc') else len(model.names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
