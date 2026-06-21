# golden/ — RTL verification vectors

Two tiers for a small fixed COCO image set:

1. **`golden/vectors/`** — the canonical **bit-exact INTEGER** goldens the
   Cognichip RTL testbench checks against bit-for-bit (`gen_golden_fixed.py`).
2. **`golden/float_ref/`** — the FP32 reference goldens, kept for comparison
   (`gen_golden.py`).

## Integer goldens (`golden/vectors/`, manifest_fixed.json)
Every Conv2d is recomputed in integer fixed-point EXACTLY as the HW requant unit:
```
Xq  = clip(round(X / act_scale_in), -127, 127)          # INT8 input
acc = conv2d(Xq, Wq)                                     # INT32 exact MAC, INT8 weights
acc += round(bias_fp32 / (w_scale[oc]·act_scale_in))     # INT32 bias
Yq  = clip(round(acc · w_scale[oc]·act_scale_in/act_scale_out), -127, 127)   # INT8 out
```
No floats in the datapath. Per image id `<id>`:
```
input_int8.npy       INT8 network input (S0 pre-processor output)
layers_int8.npz      INT8 per-Conv2d output activations  ← RTL checks these
acc_int32.npz        INT32 pre-requant accumulators
detections_int.npz   boxes:int32[x1,y1,x2,y2], score_u8:uint8, cls:int32
```
`manifest_fixed.json` carries the layer order, datapath formula, the
float-vs-fixed per-layer abs-error report, and the worst-5 layers.

Layer names match `hwgraph/hw_graph.json` and `hwconst/` exactly, so a bench
lines up DUT BRAM contents, graph nodes, and golden tensors by name.

## Float reference (`golden/float_ref/`)
FP32 intermediates from the fused model (`gen_golden.py`): `input.npy`,
`layers.npz`, `output.npz` per id + `manifest.json`. Useful to attribute any RTL
mismatch to quantization vs a datapath bug.

## Reproduce
```
python golden/gen_golden.py        --num-images 4   # float reference (→ float_ref/)
python golden/gen_golden_fixed.py  --num-images 4   # integer goldens (→ vectors/)
```
`golden/vectors/` and `golden/float_ref/vectors/` are git-ignored (large,
regenerable); the manifests are tracked.
