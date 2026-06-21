# 🕯️ Prior Attempt — genesys2 / YOLOv8-n (prune + retrain track)

> **Status:** COMPLETE & SUPERSEDED. This documents the original software track
> (project \`D:\Projects\FPGA\genesys2\`) that preceded the current SiliconYOLO /
> YOLOv10n build. It finished its 2-hour time-boxed production run and froze
> weights, but was superseded by the pretrained-YOLOv10n approach used here.
> Kept for provenance, lessons learned, and Devpost "what we tried" narrative.

**Date completed:** 2026-06-21 · **Hardware:** RTX 4060 Laptop GPU, torch 2.6.0+cu124, Win11

---

## 1. Approach: compression by structured pruning + fine-tune
Start from COCO-pretrained **YOLOv8-n**, apply **L1 structured channel pruning**
(torch-pruning, dependency-graph), then **fine-tune** to recover accuracy. Then
PTQ to INT8, freeze, and hand off to the Cognichip hardware track.

Pipeline stages (A1-A6):
- **A1** repo scaffold + git, CUDA torch pinned cu124
- **A2** baseline eval harness (pycocotools, COCO val2017, 5000 imgs)
- **A3** L1 prune (ratio 0.10) -> fine-tune (3-epoch proof -> 2h production)
- **A4** INT8 PTQ + freeze (time-boxed freeze for pipeline unblocking)
- **A6** golden vectors + demo video (HyperFrames)

## 2. Results (pycocotools, COCO val2017, FP32, imgsz 640)

| Checkpoint | Params | mAP50-95 | mAP50 | vs floor (50-95) |
|---|---|---|---|---|
| FP32 baseline (unpruned YOLOv8-n) | 3.157M | **37.37** | 52.57 | — |
| **Locked floor** (−2.0 budget) | — | **35.37** | **50.57** | — |
| Post-prune, pre-finetune | 2.209M | **0.00** | 0.00 | catastrophic (expected) |
| 3-epoch proof | 2.652M | 26.55 | 39.76 | −8.82 |
| **2h production (8 epochs)** | 2.647M | **32.06** | **46.38** | **−3.31** |

- Prune reduction: **3.157M -> 2.652M params (−16%)**, **8.80 -> 7.53 GFLOPs (−14.5%)**.
- Production run: 2 h 0 m 25 s, 8 epochs (~15 min/epoch), workers=6, clean exit.
- Trajectory still rising at epoch 8 (+0.3-0.7 mAP/epoch): floors judged reachable
  with a full **~50-epoch (~12 h)** run — deliberately not done (time-box).
- Frozen handoff: \`model/frozen/pruned_a3_frozen.pt\` (MD5 == pruned.ckpt, pruned
  architecture intact). **Time-boxed freeze** to unblock A4/A6 — below floors by design.

## 3. Engineering challenges overcome (these lessons carried forward)
1. **BN-gamma importance collapsed accuracy to 0.00** -> switched to **L1 magnitude**
   importance (graceful degradation instead of collapse).
2. **Ultralytics trainer rebuilt model from YAML**, discarding pruned channels ->
   wrote **\`PrunedDetectionTrainer\`** override that preserves the pruned structure
   (verified 2.652M params at start/train/eval, not silently un-pruned).
3. **WinError 1455 (pagefile/commit exhaustion)** at 16 dataloader workers + 38
   leaked zombie python processes -> dropped to **workers=6**, killed zombies,
   ultimately required a **reboot** to clear kernel-wedged paging-I/O processes.
4. **Stale \`args.yaml\` resurrection** on resume -> wipe run dir before relaunch.
5. **TokenRouter auth**: must use \`ANTHROPIC_AUTH_TOKEN\` (Bearer), delete stale
   \`~/.claude.json\` OAuth cache — the same recipe reused in SiliconYOLO.

## 4. Why it was superseded (the pivot rationale)
The prune+retrain path was **GPU-training-bound and could not clear the mAP floors
in a hackathon time-box** (−3.31 after 2 h; ~12 h needed). Research into pretrained
options (see \`artifacts/pretrained_yolo_research.md\`) found a strictly better base:

| | genesys2 (this attempt) | **SiliconYOLO (current)** |
|---|---|---|
| Base | YOLOv8-n, **pruned + retrained** | **YOLOv10n pretrained** (no training) |
| NMS | required (NMS hardware block) | **NMS-FREE** (block eliminated) |
| Params | 2.652M (after prune) | 2.776M (stock) |
| Final accuracy | **32.06** (below 37.37 baseline) | **37.62 INT8** (beats baseline 37.37) |
| Time cost | ~2 h training (+~12 h to clear floors) | **~0 training** (PTQ only) |
| Risk | trainer/pagefile/zombie failures | low (stock checkpoint) |

YOLOv10n is smaller-ish, **more accurate**, **NMS-free** (removes an entire RTL
block), and needs **no multi-hour training** — so SiliconYOLO reaches a frozen,
floor-beating, hardware-ready model far faster and cleaner.

## 5. What carried over to SiliconYOLO
- The **eval harness** design (pycocotools, COCO val2017, conf 0.001 / iou 0.7).
- The **freeze -> hw_graph.json + hwconst/.mem + golden vectors** handoff contract.
- The **HyperFrames demo-video** pipeline (a new YOLOv10n video is being produced).
- TokenRouter + bypass-mode Claude Code operating recipe.
- All the Windows/dataloader stability fixes (workers cap, zombie hygiene).

## 6. Artifact locations (in the genesys2 project, retained read-only)
- Production report: \`genesys2/docs/A3_production_2h_report.md\`
- Eval JSONs: \`genesys2/model/_prod_eval.json\` (32.06), \`_pruned_proof_eval.json\`
  (26.55), \`_pruneinit_eval.json\` (0.00), \`_conv_eval.json\`
- Frozen weights: \`genesys2/model/frozen/pruned_a3_frozen.pt\` (2.652M, time-boxed)
- Demo video: \`artifacts/silicon_yolo_demo.mp4\` (YOLOv8-n era)
- Git tip: \`b39ea8f A3 production (2h time-boxed) complete + weights frozen\`

*This attempt is preserved as the baseline we improved upon — not the shipping model.*

---

### 📂 Project materials
[🔬 README](../README.md) · [📝 Devpost](DEVPOST_SUBMISSION.md) · [💸 Cost analysis](COST_COMPARISON.md) · [🧪 Sim showcase](../rtl_tb/SIM_SHOWCASE.md) · [🕯️ Prior attempt](PRIOR_ATTEMPT_YOLOV8N.md) · [🎬 Demo video](../video/renders/silicon_yolo_v10n_demo_with_cost.mp4)
