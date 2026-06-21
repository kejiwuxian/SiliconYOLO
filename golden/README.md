# golden/ — bit-accurate RTL verification vectors

For a small fixed COCO image set, `gen_golden.py` dumps the reference tensors the
RTL bench (`rtl_tb/`, REF-08) checks the fixed-weight datapath against.

## Layout
```
golden/manifest.json          layer order, shapes, dtypes, provenance (the contract)
golden/vectors/<id>/input.npy   preprocessed NCHW input (fp32 [0,1], letterboxed)
golden/vectors/<id>/layers.npz  every Conv2d/SiLU/MaxPool/Upsample output, keyed by layer name
golden/vectors/<id>/output.npz  final NMS-free detections
```
Layer names match `hwgraph/hw_graph.json` and `hwconst/` exactly (captured on the
fused model), so a bench can line up DUT BRAM contents, graph nodes, and golden
tensors by name.

## Precision
These are the **FP32 reference** tensors plus the file/format contract — enough
to bring up and structurally verify the pipeline now. The bit-accurate
fixed-point golden (driven by the `requant` params in
`hwconst/quant_scales.json`) is the production follow-up and slots into the same
file layout.

## Reproduce
```
python golden/gen_golden.py --num-images 4      # CPU; ~3 s/image
python golden/gen_golden.py --dry-run           # 2 imgs, code-path check
```
`golden/vectors/` is git-ignored (large, regenerable); `manifest.json` is tracked.
