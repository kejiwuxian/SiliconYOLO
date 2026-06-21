# 🎙️ Silicon YOLO — Demo Video Narration Script

Voiceover script for **`video/renders/silicon_yolo_v10n_demo_voiceover_taalas.mp4`**
(**180 s** · 1920×1080 · 30 fps · stereo AAC). Narration generated with **ElevenLabs**
(*Brian* voice, *Eleven Multilingual v2*), timed to the HyperFrames scene boundaries.

> **Pronunciation notes for TTS:** write *"Silicon Yolo"* (title-case, not all-caps) so it
> is read as a word, and respell *"RISC-V"* as **"risk five"** — on-screen text keeps the
> canonical spellings, only the narration text is phoneticized.

---

## Part 1 — The chip (0–98 s)

### Hook  ·  `s1` @ 0.6 s
> What if an entire object detector lived inside one chip — no GPU, no cloud? This is Silicon YOLO: a pretrained neural network, frozen into silicon.

### NMS-free  ·  `s2` @ 12.6 s
> We start from YOLOv10n. Unlike older detectors, it's NMS-free — it needs no non-maximum-suppression step. So an entire, awkward hardware block simply disappears, leaving a clean feed-forward pipeline that maps beautifully onto an FPGA fabric.

### Flow  ·  `s3` @ 32.6 s
> The flow is simple. Take the pretrained weights, quantize them to eight-bit integers, then freeze them into a fixed contract — and hand that contract to hardware. No retraining, no fine-tuning loops. Just a deterministic path from model to metal.

### Results  ·  `s4` @ 48.6 s
> And the results speak for themselves. INT8 quantization is near-lossless — accuracy barely moves from the floating-point baseline. The design runs at fifty-one frames per second on a Kintex-7, using thirty-eight thousand LUTs, just three-point-two watts, and zero DSP blocks — every multiply built from constant-coefficient logic. And every datapath is verified bit-exact against the software golden vectors.

### Why  ·  `s5` @ 74.6 s
> Why does this matter? It unlocks private, milliwatt-class AI at the edge — vision that runs on a battery, with nothing ever leaving the device.

### Tagline  ·  `s6` @ 84.5 s
> Silicon YOLO. We froze a neural network into silicon.

### Cost & efficiency  ·  `cost` @ 90.6 s
> And it's economical, too — about twenty-six times better energy per frame, and roughly two dollars per chip at volume.

---

## Part 2 — The bet (98–180 s)

Four positioning scenes appended after the technical demo, in the unified dark style
(grid background, blue/amber accents). Each is a self-contained `seg_t*` clip.

### The contrarian bet  ·  `n1` @ ≈98 s
> Here's our bet. Hardcoding a model into silicon is a proven idea — Taalas just raised two hundred and nineteen million dollars doing it for small language models. We're betting on the opposite layer. The biggest edge opportunity isn't language, it's perception: vision, voice, and sensing. And a perception model never changes, which makes it perfect to freeze into a chip.

### Market potential  ·  `n2` @ ≈120 s
> And the market agrees. Non-LLM edge AI, mostly computer vision, is already a twenty-billion-dollar market, on track to cross one hundred billion dollars by twenty thirty, growing nearly thirty percent a year. The on-device language-model market is real, but several times smaller and slower. Perception is simply the larger prize, and it's exactly where fixed-function silicon wins.

### Modular by design  ·  `n3` @ ≈145 s
> We also built it to fit. Silicon Yolo is a self-contained accelerator block. Control rides on a standard AXI-Lite bus, while pixels and detections stream over AXI-Stream. So it drops straight into any system-on-chip, right next to a CPU, an image processor, or a risk five core. No glue logic required.

### The thesis (close)  ·  `n4` @ ≈166 s
> So that's the thesis. The next wave of edge AI won't run on smaller software. It will be the model itself, etched in silicon. Silicon Yolo: a drop-in vision brain for the next billion devices.

---

## Production notes

- **Total runtime:** exactly **180.000 s** (5400 frames @ 30 fps), enforced with `-t 180.000`.
- **Narration timing:** Part-2 clips are tempo-adjusted **1.07×** with phased inter-clip gaps so speech lands inside each scene without overlap (verified via `silencedetect`).
- **Single-segment fixes:** narration can be re-rendered for one scene and swapped in **audio-only** (video stream copied untouched), which preserves the exact 180 s timeline.
- **Cost slide** is composited over 90–98 s in the unified blue/amber + grid style to match Part 2.

---

<sub>📂 Project materials: [README](../README.md) · [Devpost](../docs/DEVPOST_SUBMISSION.md) · [Cost analysis](../docs/COST_COMPARISON.md)</sub>
