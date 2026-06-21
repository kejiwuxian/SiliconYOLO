#!/usr/bin/env python
"""VCD -> annotated waveform PNG (matplotlib).

A small, dependency-light VCD parser + digital-waveform renderer used to turn the
unit-testbench dumps in rtl_tb/sim_out/*.vcd into annotated PNGs for the
simulation showcase. No GTKWave required.

Renders each selected signal as a digital waveform: 1-bit signals as logic-level
step lines, multi-bit buses as a "data box" lane with the (signed or unsigned)
value printed in each segment.

Usage:
  python tools/vcd_to_png.py rtl_tb/sim_out/csd_mac_slice.vcd \
      --out artifacts/sim_waveform_csd_mac_slice.png \
      --title "CSD MAC slice — INT8 8-tap dot product (Icarus Verilog)" \
      --signals clock reset mac_en acc_clear act_i weight_i acc_q \
      --signed act_i weight_i acc_q \
      --annotate "act_i/weight_i: INT8 operands streamed in" \
      --annotate "acc_q: INT32 running accumulate (0-DSP CSD MAC)"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# --------------------------------------------------------------------------- #
# Minimal VCD parser
# --------------------------------------------------------------------------- #
def parse_vcd(path: Path):
    """Return (signals, end_time).

    signals: dict name -> dict(width:int, changes:list[(time, intval_or_None)])
    Bus values that contain x/z are stored as None.
    """
    id_to_names: dict[str, list[str]] = {}
    width: dict[str, int] = {}
    text = path.read_text(errors="replace").splitlines()

    # --- header: $var declarations ---
    i = 0
    cur_scope = []
    while i < len(text):
        line = text[i].strip()
        if line.startswith("$var"):
            # $var wire 8 ! act_i [7:0] $end   (tokens vary)
            toks = line.split()
            w = int(toks[2])
            vid = toks[3]
            nm = toks[4]
            full = ".".join(cur_scope + [nm]) if cur_scope else nm
            id_to_names.setdefault(vid, []).append(nm)
            id_to_names.setdefault(vid + "|full", []).append(full)
            width[vid] = w
        elif line.startswith("$scope"):
            toks = line.split()
            if len(toks) >= 3:
                cur_scope.append(toks[2])
        elif line.startswith("$upscope"):
            if cur_scope:
                cur_scope.pop()
        elif line.startswith("$enddefinitions"):
            i += 1
            break
        i += 1

    # --- value changes ---
    changes: dict[str, list] = {vid: [] for vid in width}
    t = 0
    end_time = 0
    val_re = re.compile(r"^([bB])([01xXzZ]+)\s+(\S+)$")
    while i < len(text):
        line = text[i].strip()
        if not line:
            i += 1
            continue
        if line[0] == "#":
            t = int(line[1:])
            end_time = max(end_time, t)
        elif line[0] in "bB":
            m = val_re.match(line)
            if m:
                bits = m.group(2)
                vid = m.group(3)
                if vid in width:
                    if any(c in "xXzZ" for c in bits):
                        changes[vid].append((t, None))
                    else:
                        changes[vid].append((t, int(bits, 2)))
        elif line[0] in "01xXzZ":
            lvl = line[0]
            vid = line[1:]
            if vid in width:
                if lvl in "xXzZ":
                    changes[vid].append((t, None))
                else:
                    changes[vid].append((t, int(lvl)))
        i += 1

    # build by short name (last var with that name wins; TB signals are unique)
    signals = {}
    for vid, w in width.items():
        names = id_to_names.get(vid, [])
        if not names:
            continue
        nm = names[0]
        signals[nm] = {"width": w, "changes": sorted(changes[vid])}
    return signals, end_time


def _to_signed(v, width):
    if v is None:
        return None
    if width <= 1:
        return v
    if v >= (1 << (width - 1)):
        return v - (1 << width)
    return v


def value_at(changes, t):
    """Last value at or before time t (None if undefined)."""
    val = None
    for (ct, cv) in changes:
        if ct <= t:
            val = cv
        else:
            break
    return val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vcd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Simulation waveform")
    ap.add_argument("--signals", nargs="+", required=True)
    ap.add_argument("--signed", nargs="*", default=[])
    ap.add_argument("--annotate", action="append", default=[])
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    signals, end_time = parse_vcd(Path(args.vcd))
    if end_time <= 0:
        end_time = 1
    signed_set = set(args.signed)

    sel = [s for s in args.signals if s in signals]
    missing = [s for s in args.signals if s not in signals]
    if missing:
        print(f"  [warn] signals not found in VCD: {missing}")
    if not sel:
        print("!! none of the requested signals were found")
        print("   available:", ", ".join(sorted(signals)[:40]))
        return 1

    n = len(sel)
    fig, axes = plt.subplots(n, 1, figsize=(15, 1.05 * n + 1.6), sharex=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#05070d")

    BLUE = "#3ba9ff"
    AMBER = "#ffb23e"
    GREEN = "#5fd391"
    GRID = "#1d2738"
    TXT = "#eaf2ff"
    MUT = "#8a9bb4"

    # dense time grid
    T = np.linspace(0, end_time, max(800, end_time + 1))

    for ax, name in zip(axes, sel):
        ax.set_facecolor("#0a1019")
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=MUT, labelsize=8)
        w = signals[name]["width"]
        ch = signals[name]["changes"]
        is_signed = name in signed_set

        if w == 1:
            ys = np.array([value_at(ch, t) if value_at(ch, t) is not None else 0 for t in T])
            ax.step(T, ys, where="post", color=BLUE, linewidth=1.8)
            ax.set_ylim(-0.3, 1.3)
            ax.set_yticks([0, 1])
        else:
            # bus lane: draw a constant-height band, print value per segment
            ax.set_ylim(-0.3, 1.3)
            ax.set_yticks([])
            col = AMBER if is_signed and ("acc" in name) else GREEN
            # segment boundaries from change times
            times = [t for (t, _) in ch] + [end_time]
            seen = set()
            for k in range(len(ch)):
                t0 = ch[k][0]
                t1 = ch[k + 1][0] if k + 1 < len(ch) else end_time
                if t1 <= t0:
                    continue
                raw = ch[k][1]
                val = _to_signed(raw, w) if is_signed else raw
                # hexagon-ish data box
                ax.add_patch(plt.Rectangle((t0, 0.12), t1 - t0, 0.76,
                                           facecolor="none", edgecolor=col, linewidth=1.4))
                label = "x" if val is None else str(val)
                if (t1 - t0) > end_time * 0.012:
                    ax.text((t0 + t1) / 2, 0.5, label, ha="center", va="center",
                            color=TXT, fontsize=8, fontfamily="monospace")

        ax.set_ylabel(name, color=TXT, fontsize=10, rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.045, 0.5)
        ax.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.5)
        ax.set_xlim(0, end_time)

    axes[-1].set_xlabel("simulation time (ps)", color=MUT, fontsize=9)

    fig.suptitle(args.title, color=TXT, fontsize=15, fontweight="bold", y=0.995)
    sub = args.subtitle or "Icarus Verilog · VCD · rtl_tb/sim_out"
    fig.text(0.5, 0.955, sub, ha="center", color=MUT, fontsize=10)

    if args.annotate:
        note = "   ·   ".join(args.annotate)
        fig.text(0.5, 0.012, note, ha="center", color=BLUE, fontsize=9)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.10, hspace=0.25)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=130, facecolor=fig.get_facecolor())
    print(f"  wrote {out}  ({n} signals, t=0..{end_time}ps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
