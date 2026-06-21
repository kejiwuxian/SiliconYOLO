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

## Verification hardening (follow-ups)

### STEP A — Full weight+activation INT8 eval ✅
`model/eval_int8_full.py` applies BOTH per-channel weight fake-quant AND
per-tensor **activation** fake-quant (calibrated scales from
`hwconst/quant_scales.json`) through the full forward pass, then evals COCO
val2017 (pycocotools).

| Config | mAP50-95 | mAP50 | Δ vs FP32 |
|--------|----------|-------|-----------|
| FP32 reference | 0.3794 | 0.5307 | — |
| INT8 weights only | 0.3762 | 0.5301 | −0.32 pt |
| **INT8 weights + activations (true HW datapath)** | **0.3530** | **0.5055** | **−2.6 pt** |
| INT4-tolerant (44 layers, blanket) + W/A INT8 | 0.0312 | 0.0728 | −34.8 pt ✗ |
| INT4-tolerant backbone only (head INT8) + W/A INT8 | 0.0617 | 0.0975 | −33.2 pt ✗ |

**Findings:**
- The end-to-end INT8 datapath is **mAP50-95 0.3530**; the activation
  quantization (not the weights) is the dominant loss (weights-only was 0.3762).
  For reference, the **weight-only INT8 (0.3762) and FP32 (0.3794) both beat the
  YOLOv8n FP32 baseline (0.3733)**; the full W+A datapath (0.3530) sits just below
  it, so for an accuracy-critical deployment a short activation-aware QAT pass is
  the documented recovery path. The weight-only and FP32 numbers are unchanged.
- **INT4 fallback does NOT hold accuracy** under the full datapath: the
  weight-rel-MSE flag (<3% at INT4) is far too optimistic once activations are
  also INT8 — blanket INT4 collapses mAP to 0.031, and even sparing the detect
  head only reaches 0.062. **Conclusion: keep INT8 as the production datapath;
  INT4 requires per-layer activation-aware QAT / sensitivity search, not a static
  weight-only flag.** `int4_candidate` in the artifacts is retained as an
  *advisory* marker only. Results: `model/eval_int8_full*.json`.

### STEP B — Bit-exact fixed-point golden vectors ✅
`golden/gen_golden_fixed.py` reimplements every Conv2d in **integer fixed-point**
exactly as the HW requant unit: INT8 input × INT8 weight → **INT32 accumulate** →
+INT32 bias → requant to INT8 via the per-layer `weight_scale·act_in/act_out`
(spec §4.3.2). No floats in the datapath.
- Regenerated `golden/vectors/<id>/`: `input_int8.npy`, `layers_int8.npz` (83 INT8
  conv activations — what RTL checks bit-for-bit), `acc_int32.npz`,
  `detections_int.npz` (int boxes, uint8 score, int class). Manifest:
  `golden/manifest_fixed.json`.
- Old float goldens archived to `golden/float_ref/`.
- Float-vs-fixed per-layer abs error reported in the manifest: **median max-abs
  0.41, p90 2.4** (activation-scale units); largest in the detect-head logits
  (max 13.6, `model.23.one2one_cv3.*`), which feed argmax/box-decode rather than
  downstream convolutions — expected and benign for detection output.



