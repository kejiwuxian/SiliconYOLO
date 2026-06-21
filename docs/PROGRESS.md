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

## STEP 3 — FP32 baseline eval ✅
- `model/eval_coco.py` over full COCO val2017 (5000 imgs), ultralytics validator
  (faster-coco-eval / pycocotools), GPU.
- **FP32: mAP50-95 = 0.3794, mAP50 = 0.5307** (`model/eval_fp32.json`).
  Matches official YOLOv10n (~38.5). This is the reference; nothing to "recover"
  since we did not prune/train.
- Note: ultralytics `val()` auto-fuses → eval runs on the fused (BN-folded,
  NMS-free one2one head) model = **83 Conv2d / 2.299 M params**. This fused model
  is the canonical hardware target.

## STEP 4 — INT8 PTQ + calibration ✅
- `quant/ptq.py` on the fused model: per-channel symmetric INT8 weight scales +
  per-tensor activation calibration over **256** COCO val images.
- `hwconst/quant_scales.json`: 83 Conv2d layers, **44 flagged INT4-tolerant**
  (rel weight-MSE at INT4 < 3%).
- **INT8 (weight-quant) eval: mAP50-95 = 0.3762, mAP50 = 0.5301**
  (`model/eval_int8.json`). Drop vs FP32 = **−0.0032 mAP50-95 (−0.32 pt)** —
  essentially lossless. (This mAP isolates INT8 *weight* quantization; activation
  scales are calibrated and exported for the HW datapath, full
  weight+activation fake-quant in the eval loop is a documented follow-up.)

## STEP 5 — Freeze weights ✅
- `model/freeze.py` froze the fused, NMS-free YOLOv10n (no training → immediate):
  - `model/frozen/yolov10n_fused_fp32.pt` — fused FP32 reference state_dict.
  - `model/frozen/yolov10n_int8_frozen.pt` — **stable path**: per-layer INT8
    integer weights + per-channel scales + FP32 biases (83 Conv2d layers).
  - `model/frozen/FROZEN.json` — manifest (sha256, provenance, no_training=true).

## STEP 6 — Hardware handoff artifacts ✅
- `hwgraph/hw_graph.json` — ordered HW contract (spec §4.3.1): 83 Conv2D layers,
  per-layer op/c_in/c_out/kernel/stride/padding/groups/activation/quant bits/
  requant + the `.mem`/`.coe` filenames. **`"nms": false`** (no NMS RTL block).
- `hwconst/` — **83 weight + 82 bias** `.mem`+`.coe` pairs under `hwconst/mem/`
  (INT8 weights, INT32 biases, MSB-first hex, `[c_out][c_in][kh][kw]`), plus
  `quant_scales.json` (enriched with spec §4.3.2 requant map) and `index.json`.
  Formats documented in `hwconst/README.md`. (`model.23.dfl.conv` is param-free →
  no bias file.)
- `golden/` — `gen_golden.py` produced **4 golden vectors** (input + 90 per-layer
  intermediate tensors + final detections each) + `manifest.json`. FP32 reference
  + bit-accurate file/format contract for `rtl_tb/`.

### ✅ Cognichip RTL is UNBLOCKED
All three handoff artifacts exist and align with `yolov10n_accel_spec.md`
(REF-04/05/06/08): `hwgraph/hw_graph.json`, `hwconst/` (.mem/.coe +
quant_scales.json), and `golden/` vectors.


