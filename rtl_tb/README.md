# rtl_tb/ — RTL testbench scaffolding (placeholder)

Reserved for the RTL testbench that verifies the fixed-weight datapath against
the **golden vectors** in `golden/vectors/`. The flow:

1. Load `hwconst/*.mem` (or `.coe`) into the BRAM/ROM weight init of the DUT.
2. Drive `golden/vectors/<id>/input.npy` into the pipeline.
3. Compare per-layer DUT outputs against `golden/vectors/<id>/layers.npz` and the
   final detections against `output.npz`, within the INT8 tolerance from
   `hwconst/quant_scales.json`.

The contract (layer order, shapes, dtypes) is `golden/manifest.json` +
`hwgraph/hw_graph.json`.
