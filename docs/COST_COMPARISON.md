# 💸 Silicon YOLO — Cost & Efficiency Comparison
### Taped-out fixed-weight ASIC vs. traditional hardware

> **Scope:** This compares the **taped-out fixed-weight ASIC** (the real "weights baked into
> silicon" product) against off-the-shelf hardware running the *same* detection task —
> YOLOv10n-class, 640×640, INT8, ~37 mAP50-95. The Genesys 2 / Kintex-7 FPGA is **only the
> functional prototype** used to prove the RTL; it is *not* the product and is excluded here.
> All ASIC figures are **engineering estimates** with assumptions stated at the bottom.

---

## TL;DR

| Metric | Silicon YOLO ASIC | Best traditional option | Advantage |
|---|---|---|---|
| **Power** | ~0.20 W (200 mW) | Coral 2.0 W | **~10× lower** |
| **Energy / inference** | ~3.9 mJ/frame | Hailo-8 25 mJ | **~6× better** (≥26× vs Jetson, 73× vs GPU) |
| **Efficiency** | ~255 FPS/W | Hailo-8 40 FPS/W | **~6× better** |
| **Unit cost @100k vol** | ~$2 | Pi 5 $80 / Coral $130 | **40–150× cheaper silicon** |
| **3-yr fleet TCO (100k, 24/7)** | ~$2.78 M | Pi 5 $10.8 M | **~4–27× lower** |
| **Accuracy (mAP)** | 37.6 | 36–37.4 | **≈ equal** (same frozen weights) |

![ASIC vs edge hardware — power, energy, cost, TCO](showcase/silicon_yolo_asic_cost_comparison.png)

The ASIC trades **flexibility** (one model, hard-wired) for a **decisive win on power, energy,
unit cost, and total cost of ownership at volume**.

---

## Why the ASIC is so much cheaper & cooler than an FPGA or GPU

The design is *fixed-weight*: every YOLOv10n weight is realised as a **CSD (canonical-signed-digit)
constant multiplier baked directly into the logic** — there are **no DSP blocks, no weight SRAM, and
no external DRAM**. That has three compounding effects when moved from FPGA fabric to a real ASIC:

- **No reconfigurable fabric tax.** FPGA LUTs/routing burn ~10–20× the power and area of equivalent
  hard logic. The 3.2 W the design draws on Kintex-7 collapses to **~0.2 W on a 28 nm ASIC** at the
  same 51 FPS — and far less when clock-gated or duty-cycled.
- **Weights cost ~zero area/energy.** Hard-wired constants don't need to be fetched, stored, or
  multiplied by a general multiplier. No DRAM traffic = no off-chip energy (usually the dominant
  cost in edge inference).
- **Tiny die.** ~5 mm² on 28 nm (logic + on-chip activation SRAM), which is what drives the
  ~$2/unit silicon cost at volume.

> **⏱️ Two clock modes, same efficiency.** Because the datapath is logic-only (0 DSP, all CSD
> shift-add), the same netlist closes timing ~3–4× faster off the FPGA fabric (the Kintex-7 is
> itself a 28 nm part, so this is a same-node fabric-overhead win). That lets the ASIC clock to
> **~800 MHz → up to ~200 FPS** for throughput-bound jobs, or stay at **~200 MHz / ~51 FPS /
> ~0.2 W** milliwatt-class for battery/always-on. Throughput *and* power scale ~linearly with
> clock, so **efficiency is ~constant at ~255 FPS/W** at either point. **Every cost figure below
> uses the low-power (milliwatt) point** — the headline product mode.

---

## Core comparison

| Platform | Type | Power | FPS @640 INT8 | mAP50-95 | Unit cost (@100k) | NRE |
|---|---|---|---|---|---|---|
| Silicon YOLO ASIC (28nm est.) | Fixed-weight ASIC | 200 mW | 51 | 37.6 | $2 | $2.5M |
| Jetson Orin Nano Super | Edge GPU SoC | 15 W | 150 | 37.3 | $249 | -- |
| Hailo-8 (accel+host) | NN accelerator | 2 W | 100 | 37.0 | $200 | -- |
| Coral Edge TPU (dev board) | NN accelerator | 2 W | 35 | 36.0 | $130 | -- |
| Desktop RTX 4060 | Desktop GPU | 115 W | 400 | 37.4 | $300 | -- |
| Raspberry Pi 5 (CPU only) | CPU SBC | 7 W | 2 | 37.3 | $80 | -- |

*Same task, same frozen INT8 weights → accuracy is essentially identical across platforms; the
differentiator is power, cost, and determinism, not mAP.*

---

## Efficiency (the headline win)

| Platform | FPS/W (efficiency) | Energy / inference | FPS/$ |
|---|---|---|---|
| Silicon YOLO ASIC (28nm est.) | 255.0 | 3.9 mJ | 25.500 |
| Jetson Orin Nano Super | 10.0 | 100.0 mJ | 0.602 |
| Hailo-8 (accel+host) | 40.0 | 25.0 mJ | 0.500 |
| Coral Edge TPU (dev board) | 17.5 | 57.1 mJ | 0.269 |
| Desktop RTX 4060 | 3.5 | 287.5 mJ | 1.333 |
| Raspberry Pi 5 (CPU only) | 0.3 | 3500.0 mJ | 0.025 |

- **Energy/frame:** Silicon YOLO ASIC ≈ **3.9 mJ** vs Jetson Orin Nano **100 mJ** (**26×**),
  Desktop RTX 4060 **287 mJ** (**73×**), Hailo-8 **25 mJ** (**6×**).
- **FPS/W:** **255** for the ASIC vs 40 (Hailo), 10 (Jetson), 3.5 (GPU).

---

## Total cost of ownership (the volume win)

3-yr fleet TCO  (100,000 units, 24/7 = 26280 h, $0.15/kWh):

| Platform | Hardware capex | 3-yr energy | Total 3-yr TCO | vs ASIC |
|---|---|---|---|---|
| Silicon YOLO ASIC (28nm est.) | $2,700,000 | $78,840 | $2,778,840 | 1.0x |
| Jetson Orin Nano Super | $24,900,000 | $5,913,000 | $30,813,000 | 11.1x |
| Hailo-8 (accel+host) | $20,000,000 | $985,500 | $20,985,500 | 7.6x |
| Coral Edge TPU (dev board) | $13,000,000 | $788,400 | $13,788,400 | 5.0x |
| Desktop RTX 4060 | $30,000,000 | $45,333,000 | $75,333,000 | 27.1x |
| Raspberry Pi 5 (CPU only) | $8,000,000 | $2,759,400 | $10,759,400 | 3.9x |

- **Break-even volume (hardware capex, ASIC vs Jetson Orin Nano): ~10,100 units** — above this volume the ASIC is cheaper than buying Jetsons on
  **hardware capex alone**; once 24/7 power is included, the crossover happens far sooner.
- The ~$2.5 M **NRE** (mask set + design/verification/bring-up on 28 nm) is the price of entry.
  It is amortised away by ~10 k units and becomes a rounding error at 100 k+.

---

## Other metrics (beyond cost)

| Metric | Silicon YOLO ASIC | GPU / SoC / accelerator |
|---|---|---|
| **Latency** | ~20 ms/frame, **fully deterministic** (fixed pipeline, no OS, no driver/runtime jitter) | Variable — OS scheduling, driver, batching, thermal throttling |
| **Form factor** | Single small die (~5 mm²), QFN/WLCSP; integrable into a sensor module | SoM / M.2 card / PCIe GPU — board + heatsink |
| **Thermal** | **Passive** (~0.2 W); no fan/heatsink | Jetson active-cooled; GPU needs fans; Hailo/Coral mild heatsink |
| **Boot / availability** | Instant-on, no OS to boot, no model load | Seconds to boot OS + load model into runtime |
| **Supply-chain / security** | Self-contained; weights physically in silicon (can't be exfiltrated or swapped) | Model file on flash — copyable/tamperable |
| **BOM beyond compute** | Sensor + tiny MCU/PMIC | Sensor + full host SoC + DRAM + storage + cooling |
| **Flexibility** ⚠️ | **One model, frozen.** No retraining, no new classes without a re-spin | **Fully reprogrammable** — swap models/weights in software |
| **Time-to-change** ⚠️ | Months (new tape-out) | Minutes (push a new model file) |

---

## When the ASIC wins — and when it doesn't

**✅ Wins decisively when:**
- The model is **stable** and the deployment is **single-purpose** (e.g. a smart-camera SKU,
  doorbell, retail counter, drone payload, automotive sensor).
- **Volume is high** (≳10 k units) and/or devices run **always-on** (battery/solar/PoE budgets).
- **Ultra-low power, passive cooling, deterministic latency,** or **on-die model security** matter.

**❌ Loses when:**
- You need to **change models often**, support **many models**, or are pre-product-market-fit.
- **Volume is low** (<~5–10 k) — NRE never amortises; just buy a Jetson/Hailo.
- You need accuracy beyond a nano-class detector (the silicon is sized for this one network).

> **Strategic read:** prototype and prove on the **Kintex-7 FPGA** (already done — bit-exact to the
> golden vectors); commit to the **ASIC** only once the model is frozen and volume is committed.
> That is exactly the path this project follows.

---

## Assumptions & caveats (so the numbers are auditable)

- **ASIC process:** 28 nm, ~5 mm² die (logic + ~17 Mb on-chip activation SRAM; weights are CSD
  constants, **0 DSP, no external DRAM**). Unit cost ~$2 at **100 k+** volume (wafer + package +
  test, ~90% yield). **NRE ~$2.5 M** (28 nm mask set + design/PD/verification/bring-up).
- **ASIC power ~0.2 W** derived from the FPGA's measured 3.2 W logic ÷ ~15× fabric-vs-ASIC factor,
  at iso-performance (51 FPS). Drops to tens of mW when duty-cycled. The logic-only datapath also
  closes timing ~3–4× faster off fabric, so the same chip can instead clock to **~800 MHz → ~200 FPS**
  at ~0.8 W (same ~255 FPS/W efficiency); the low-power 200 MHz / 51 FPS point is used throughout for the cost math.
- **Comparators** are the *inference engine* of each platform at YOLOv8n/v10n, 640², INT8:
  Jetson Orin Nano Super ($249, 67 TOPS, ~150 FPS, ~15 W typ), Hailo-8 (~$200, 26 TOPS, ~100 FPS,
  ~2.5 W — needs a host), Coral Edge TPU dev board (~$130, 4 TOPS, ~35 FPS, ~2 W),
  Desktop RTX 4060 (~$300 GPU, ~400 FPS, ~115 W), Raspberry Pi 5 CPU (~$80, ~2 FPS, ~7 W).
- **TCO:** 100,000 units, 24/7 for 3 years (26,280 h) at $0.15/kWh; hardware capex + energy only
  (excludes integration, cooling, failures, networking).
- Prices/throughput are approximate 2024–25 figures and vary with vendor, batch size, and tuning.
  Treat all ASIC figures as **order-of-magnitude engineering estimates**, not a foundry quote.

---

*Chart: [`showcase/silicon_yolo_asic_cost_comparison.png`](showcase/silicon_yolo_asic_cost_comparison.png) · Generator: `tools/asic_cost_compare.py`*

---

### 📂 Project materials
[🔬 README](../README.md) · [📝 Devpost](DEVPOST_SUBMISSION.md) · [💸 Cost analysis](COST_COMPARISON.md) · [🧪 Sim showcase](../rtl_tb/SIM_SHOWCASE.md) · [🕯️ Prior attempt](PRIOR_ATTEMPT_YOLOV8N.md) · [🎬 Demo video](../video/renders/silicon_yolo_v10n_demo_voiceover_taalas.mp4)