# video/ — Silicon YOLO (YOLOv10n) demo

HyperFrames composition for the **YOLOv10n** Silicon YOLO build. ~90 s, 1920×1080,
30 fps. Dark/premium brand (near-black `#05070d`, electric blue `#3BA9FF`, amber
`#FFB23E`, Inter, GSAP eases, animated number counters) — carried over from the
prior genesys2/YOLOv8n demo and updated for this project.

## Story (6 scenes)
1. **Hook** — "Silicon YOLO": a fixed-weight YOLOv10n detector baked into a chip.
2. **Why YOLOv10n** — NMS-free, end-to-end → an entire non-max-suppression
   hardware block is eliminated; smaller + more accurate than YOLOv8n.
3. **The flow** — pretrained YOLOv10n → prune-free PTQ → freeze → auto-generated
   hardware contract (hw_graph.json + .mem ROMs + golden vectors) → Cognichip RTL.
4. **Results** — accuracy + hardware (animated counters):
   - FP32 mAP50-95 **37.94** → INT8 **37.62** (~lossless; beats v8n FP32 37.37)
   - **1024** INT8 CSD constant-coefficient MACs · **0 DSPs** · **~38K LUT (11.7%)**
     · **~483 BRAM (57.5%)** · **~51 FPS** · **~3.2 W** (Genesys 2 / Kintex-7 XC7K325T)
   - Bit-exact fixed-point golden-vector verification seal.
5. **Why it matters** — milliwatt-class, private, on-device edge AI.
6. **Close** — "Weights as silicon. Zero DSPs. ~51 FPS. ~3.2 W."

## Build
```bash
cd video
npx hyperframes@0.6.115 lint       # 0 errors
npx hyperframes@0.6.115 validate   # 0 errors
npx hyperframes@0.6.115 render --out renders/silicon_yolo_v10n_demo.mp4
```
Final MP4 is also copied to the SimularFiles artifacts dir as
`silicon_yolo_v10n_demo.mp4`.

Assets `cover.png` / `agents.png` reused from the genesys2 video project.
