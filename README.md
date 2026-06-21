# Silicon YOLO

Fixed-weight YOLO object-detection chip for **Digilent Genesys 2 / Xilinx
Kintex-7 XC7K325T**. Weights are baked into the FPGA fabric as **CSD /
constant-coefficient multipliers (0-DSP goal)**, INT8 (INT4 for tolerant layers),
640×640, 80-class COCO.

This repository takes an **off-the-shelf pretrained model straight to a hardware
handoff** — no training. We use the official Ultralytics **YOLOv10n** COCO
checkpoint (NMS-free → no NMS RTL block), quantize it to INT8, freeze the
weights, and emit the artifacts the Cognichip RTL track needs.

## Pipeline (model → silicon handoff)

```
model/   fetch pretrained YOLOv10n        -> model/yolov10n.pt
model/   FP32 baseline eval (pycocotools) -> model/eval_fp32.json
quant/   INT8 PTQ + calibration           -> hwconst/quant_scales.json, model/eval_int8.json
model/frozen/  freeze quantized weights   -> model/frozen/yolov10n_int8_frozen.pt
hwgraph/ lower to HW op-graph             -> hwgraph/hw_graph.json
hwconst/ per-layer weights/biases         -> hwconst/*.mem, *.coe, quant_scales.json
golden/  bit-accurate test vectors        -> golden/vectors/, golden/manifest.json
```

## Reproduce

Uses the genesys2 project's CUDA PyTorch venv and its COCO val2017 (no
re-download). All commands run from the repo root.

```bash
PY=D:/Projects/FPGA/genesys2/.venv/Scripts/python.exe

$PY model/fetch_base.py                       # STEP 2  fetch pretrained base
$PY model/eval_coco.py --weights model/yolov10n.pt --tag fp32   # STEP 3 FP32 mAP
$PY quant/ptq.py --calib-images 256           # STEP 4  INT8 scales -> hwconst/
$PY model/eval_coco.py --int8 --tag int8      # STEP 4  INT8 mAP
$PY model/freeze.py                           # STEP 5  freeze weights
$PY hwgraph/export_graph.py                   # STEP 6  hw_graph.json
$PY hwconst/export_consts.py                  # STEP 6  .mem/.coe
$PY golden/gen_golden.py --num-images 4       # STEP 6  golden vectors
```

## Layout
- `model/`   — pretrained base, common helpers, FP32/INT8 eval, freeze
- `quant/`   — INT8 post-training quantization + calibration
- `hwgraph/` — `hw_graph.json` (the HW op-graph contract)
- `hwconst/` — per-layer INT8 weights/biases (`.mem`/`.coe`) + `quant_scales.json`
- `golden/`  — golden input/intermediate/output vectors for RTL verification
- `rtl_tb/`  — RTL testbench scaffolding (consumes golden vectors)
- `fw/`       — firmware / host glue
- `docs/`    — `MODEL_CHOICE.md`, `IMPLEMENTATION_PLAN.md`, `PROGRESS.md`

See `docs/IMPLEMENTATION_PLAN.md` for the single source of truth.
