# hwconst/ — fixed weight constants for the fabric

Per-layer INT8 weights + INT32 biases for the fused, NMS-free YOLOv10n, exported
as BRAM/ROM init files. These are the constants the Cognichip RTL bakes into the
fabric as CSD / constant-coefficient multipliers (0-DSP goal). Consumed by
REF-05 / REF-06 of `yolov10n_accel_spec.md`.

## Files
- `quant_scales.json` — the quant source of truth:
  - `layers.<name>.weight` — per-output-channel symmetric INT8 weight scales,
    plus INT4 candidacy (`int4_candidate`, `rel_mse_int8/4`).
  - `layers.<name>.activation` — per-tensor symmetric INT8 activation scale
    (calibrated min/max over 256 COCO val images).
  - `requant.<name>` — spec §4.3.2 view: `weight_scale`, `weight_scale_per_channel`,
    `act_scale_in`, `act_scale_out`, `shift_bits` (filled by `hwgraph/export_graph.py`).
- `index.json` — maps each layer → its `.mem`/`.coe` files, depths, shapes.
- `mem/layer_NNN_<name>_w.{mem,coe}` — INT8 weights, one per Conv2D layer.
- `mem/layer_NNN_<name>_b.{mem,coe}` — INT32 biases (absent for the param-free
  DFL conv `model.23.dfl.conv`).

The layer ids `NNN` and the exact filenames are declared by
`hwgraph/hw_graph.json` (the authority); this directory writes precisely those
files.

## `.mem` / `.coe` conventions (spec §4.3.3)
- **Format:** ASCII hexadecimal, one value per line, **MSB-first**.
- **Width:** weights = 8 bits/line (2 nibbles, two's-complement INT8);
  biases = 32 bits/line (8 nibbles, two's-complement INT32).
- **Flatten order:** `[c_out][c_in][kh][kw]` — output-channel major, row-major.
- `.mem` is for Verilog `$readmemh`; `.coe` is the Xilinx COE form
  (`memory_initialization_radix=16; memory_initialization_vector= …;`) for BRAM
  IP init. Both encode identical data.

## Quantization math (the requant contract)
```
acc_int32 = Σ (w_int8 · x_int8)            # MAC, per output channel
acc_int32 += bias_int32                     # bias_int32 = round(bias_fp32 / (weight_scale[oc]·act_scale_in))
y_int8 = clip(round(acc_int32 · weight_scale·act_scale_in / act_scale_out), -127, 127)
```
`shift_bits` is the power-of-two approximation of the combined requant multiplier
(`round(-log2(weight_scale·act_scale_in/act_scale_out))`, clamped to ≤15 per
`P_QUANT_SHIFT_MAX`); `weight_scale_per_channel` is the exact per-channel value
if the datapath prefers a true multiply over the shift.

## Reproduce
```
python quant/ptq.py --calib-images 256     # quant_scales.json
python model/freeze.py                      # model/frozen/yolov10n_int8_frozen.pt
python hwgraph/export_graph.py              # hw_graph.json + requant enrichment
python hwconst/export_consts.py             # mem/*.{mem,coe} + index.json
```
