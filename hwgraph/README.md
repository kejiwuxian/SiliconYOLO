# hwgraph/ — the hardware op-graph contract

`hw_graph.json` is the **single authority** for the network the RTL implements
(REF-04 in `yolov10n_accel_spec.md`). It is produced from the fused, NMS-free
YOLOv10n by tracing one inference in execution order.

## Per-layer schema (spec §4.3.1 + detail)
Each Conv2D entry carries:
- `id`, `name`, `op` (`"Conv2D"`)
- `c_in`, `c_out`, `kernel`, `stride`, `padding`, `groups`, `depthwise`
- `activation` (`"SiLU"` or `"none"`)
- `quant_bits_weights` (8 — the frozen precision), `quant_bits_act` (8)
- `int4_candidate` — advisory flag for the RTL `P_INT4_LAYER_MASK`
  (opportunistic 4-bit for tolerant layers; the frozen `.mem` weights are INT8)
- `in_shape`, `out_shape`
- `weight_mem_file`, `bias_mem_file` — the exact `hwconst/mem/…` files to load
- `requant` — `weight_scale`, `weight_scale_per_channel`, `act_scale_in`,
  `act_scale_out`, `shift_bits`

Structural ops (SiLU, MaxPool, Upsample, Concat boundaries) appear as lightweight
context entries (op + shapes, no id) between the Conv2D layers.

## Key properties
- **`"nms": false`** — YOLOv10n is end-to-end; the graph terminates at the detect
  head. There is **no NMS block** to build in RTL.
- 83 Conv2D layers, fused (BN folded into conv), 2.299 M params.
- Quant scheme: per-channel symmetric INT8 weights, per-tensor symmetric INT8
  activations.

## Reproduce
```
python hwgraph/export_graph.py     # writes hw_graph.json, enriches quant_scales.json
```
Run after `quant/ptq.py` (needs the calibrated scales).
