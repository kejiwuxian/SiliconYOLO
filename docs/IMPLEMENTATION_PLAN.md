# Silicon YOLO — Implementation Plan (single source of truth)

**Thesis:** skip multi-hour training by starting from an **off-the-shelf
pretrained COCO model**, then go straight to quantize → freeze →
hardware-handoff artifacts. This unblocks the Cognichip RTL track fast.

**Target:** Genesys 2 / Kintex-7 XC7K325T · INT8 (INT4 tolerant layers) ·
fixed weights as CSD/constant multipliers (0-DSP goal) · 640×640 · 80-class COCO.

**Base model:** YOLOv10n (NMS-free) — see [MODEL_CHOICE.md](MODEL_CHOICE.md).

---

## Steps & status

| Step | What | Output | Status |
|------|------|--------|--------|
| 0 | Environment (reuse genesys2 torch+COCO) | torch CUDA ✓, ultralytics ✓, COCO 5000 imgs ✓ | ✅ done |
| 1 | Repo scaffold + git | dirs, `.gitignore`, first commit | ✅ done |
| 2 | Fetch pretrained YOLOv10n | `model/yolov10n.pt`, `MODEL_CHOICE.md` | ✅ done |
| 3 | FP32 baseline eval | `model/eval_fp32.json` — **mAP50-95 0.3794** | ✅ done |
| 4 | INT8 PTQ + calibration | `hwconst/quant_scales.json`, `model/eval_int8.json` — **mAP50-95 0.3762** | ✅ done |
| 5 | Freeze weights | `model/frozen/yolov10n_int8_frozen.pt` | ✅ done |
| 6 | HW handoff artifacts | `hwgraph/hw_graph.json`, `hwconst/*.mem`/`.coe`, `golden/` | ✅ done |
| 7 | Docs + commits + report | this file, `PROGRESS.md` | ✅ done |

## Key decisions
- **YOLOv10n over YOLO11n/v8n:** NMS-free head removes an entire iterative,
  data-dependent RTL block — worth more than +1 mAP for a fixed-weight chip.
- **PTQ, not QAT:** no training budget; static PTQ with per-channel weight scales
  + per-tensor activation calibration is the minimum a fixed-point/CSD datapath
  needs. Brevitas/QONNX is the documented upgrade path (research memo §2) but is
  not required to produce a correct, frozen INT8 handoff today.
- **Reuse genesys2 assets:** its CUDA PyTorch `.venv` and `datasets/coco` are
  reused verbatim — zero re-download, RTX 4060 confirmed.

## Quantization scheme (the HW contract)
- Weights: **per-output-channel symmetric INT8**, `scale[oc] = max|W[oc]| / 127`.
- Activations: **per-tensor symmetric INT8**, calibrated min/max over a COCO val
  subset, `scale = max|range| / 127`, zero-point 0.
- INT4 candidates flagged per-layer (low dynamic range tolerant layers).

## Handoff artifacts (what unblocks RTL)
1. `hwgraph/hw_graph.json` — ordered layer list: op type, in/out channels,
   kernel, stride, padding, activation, per-layer quant params. The HW contract.
2. `hwconst/` — per-layer INT8 weights + biases as `.mem` and `.coe` (BRAM/ROM
   init) + `quant_scales.json`; formats documented in `hwconst/README.md`.
3. `golden/` — input + per-layer intermediate tensors + final detections for a
   fixed image set; bit-accurate RTL verification vectors.

## Constraints
- **No multi-hour training.** Pretrained → quantize → freeze only.
- Reuse genesys2 dataset/torch.
