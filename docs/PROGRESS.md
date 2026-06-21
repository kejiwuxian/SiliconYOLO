# Silicon YOLO — Progress Log

Chronological log of what ran and what it produced. Newest at the bottom.

## STEP 0 — Environment ✅
- Reused genesys2 venv: `D:\Projects\FPGA\genesys2\.venv\Scripts\python.exe`.
- `torch 2.6.0+cu124`, CUDA **available**, device **NVIDIA GeForce RTX 4060 Laptop GPU**.
- `ultralytics 8.4.72`, `pycocotools` OK, `onnx` present.
- COCO val2017 reused from `D:\Projects\FPGA\genesys2\datasets\coco` — **5000** images,
  `annotations/instances_val2017.json` present. No re-download.

## STEP 1 — Scaffold + git ✅
- Dirs: `model/ quant/ hwgraph/ golden/ hwconst/ rtl_tb/ fw/ docs/`.
- `.gitignore` tracks frozen artifacts + hwconst; ignores venv/datasets/__pycache__/
  Vivado out / raw `*.pt` caches / TokenRouter node scaffolding.
- `model/common.py` — shared paths, letterbox, model loader, calib iterator. Verified.

## STEP 2 — Pretrained base ✅
- `model/fetch_base.py` downloaded official **YOLOv10n** (`yolov10n.pt`, v8.4.0 asset).
- **2.776 M params · 108 Conv2d · 80 classes · `end2end=True` (NMS-free)**.
- `docs/MODEL_CHOICE.md` written (YOLOv10n vs YOLO11n/v8n justification).
