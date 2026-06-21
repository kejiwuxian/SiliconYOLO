# Model Choice — Silicon YOLO base

**Decision: YOLOv10n (official Ultralytics COCO checkpoint), used pretrained, no training.**

Target hardware: Digilent Genesys 2 / Xilinx Kintex-7 **XC7K325T**. INT8 (INT4 for
tolerant layers), **fixed weights baked into the fabric** (CSD / constant-coefficient
multipliers, 0-DSP goal), 640×640, 80-class COCO.

## Why YOLOv10n — and not YOLO11n or YOLOv8n

The single most important property for a *fixed-weight silicon* design is
**YOLOv10's NMS-free, end-to-end head**. Confirmed on the loaded checkpoint:
`DetectionModel.end2end == True`.

| Model    | Params | GFLOPs | mAP50-95 | NMS in RTL? |
|----------|--------|--------|----------|-------------|
| YOLOv8n  | 3.2 M  | 8.7    | 37.3     | **Yes** — needs an NMS block |
| **YOLOv10n** | **2.3 M** | **6.8** | **38.5** | **No** — NMS-free end-to-end |
| YOLO11n  | 2.6 M  | 6.5    | 39.5     | **Yes** — needs an NMS block |

*(Official Ultralytics figures. Our loaded checkpoint reports 2.776 M parameters
and 108 `Conv2d` layers — the count includes the dual-assignment one-to-one head
that Ultralytics fuses to a single NMS-free branch at inference.)*

### The hardware argument (decisive)
Non-Max Suppression is **iterative, data-dependent, and sequential** — sorting by
score, computing pairwise IoU, and greedily suppressing. In a fixed-weight,
streaming-dataflow chip this is the *worst* kind of block: it has no weights to
fold, needs sort/compare logic + scratch memory, and breaks the clean
feed-forward pipeline the rest of the network enjoys. YOLOv10's consistent
dual-assignment training lets the network emit final boxes directly, so we
**delete an entire RTL block** and keep the datapath purely convolutional.

For a chip whose differentiator is *weights-as-CSD-constant-multipliers, 0 DSP*,
removing NMS is worth more than YOLO11n's +1.0 mAP: every block we don't have to
build is silicon we don't have to verify, time, and power.

### Why not YOLO11n
Higher accuracy (39.5 vs 38.5 mAP) and slightly lower FLOPs, but it **keeps the
NMS post-processing block**. The +1.0 mAP does not justify adding sequential,
data-dependent sort/IoU hardware to an otherwise fully-streaming fixed-weight
pipeline. If accuracy ever becomes the hard constraint, YOLO11n is the drop-in
fallback (same Ultralytics tooling, same flow) — but the current optimum for a
0-DSP fixed-weight chip is NMS-free.

### Why not YOLOv8n (the genesys2 MVP base)
v8n was the prior project's base because a prune+retrain pipeline already
targeted it. Here we have **no training budget** and start from a pretrained
checkpoint, so we are free to pick the architecturally-best base — and v10n is
strictly better than v8n on params, FLOPs, accuracy, *and* the NMS-free property.

## Specs (this checkpoint)
- Source: `github.com/ultralytics/assets` → `yolov10n.pt` (v8.4.0 release asset)
- Params: **2.776 M** · Conv2d layers: **108** · classes: **80** (COCO)
- Input: 640×640 RGB, letterboxed (pad value 114)
- Head: **NMS-free** end-to-end (`end2end=True`)
- Expected accuracy (official): **mAP50-95 ≈ 38.5**, mAP50 ≈ 53.2

## Implication for the rest of the pipeline
- **hwgraph/**: no NMS node in `hw_graph.json` — the graph terminates at the
  end-to-end detect head; downstream is just box decode + top-k.
- **quant/**: standard per-channel weight / per-tensor activation INT8 PTQ over
  the convolutional backbone+neck+head.
- **rtl_tb/ + golden/**: golden vectors stop at the raw detections (no NMS
  reference needed), simplifying bit-accurate RTL verification.
