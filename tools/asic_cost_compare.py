import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np, json

# ---- Assumptions (clearly-labelled estimates) ----
HOURS_3YR = 3*365*24            # 26,280 h always-on
KWH_RATE  = 0.15               # $/kWh (US commercial avg ~2024-25)
FLEET     = 100_000            # units in TCO scenario

# Same detection task across all rows: YOLOv8n/v10n class, 640x640, INT8, ~37 mAP.
# "Silicon YOLO ASIC" = the taped-out fixed-weight chip (28nm est.), NOT the FPGA proto.
plats = [
 # name,                         unit$,   NRE$,     power_W, fps,  mAP,  note
 ("Silicon YOLO\nASIC (28nm est.)", 2.0,  2_500_000, 0.20,   51,  37.6, "fixed: 1 model baked in"),
 ("Jetson Orin\nNano Super",       249.0,    0,      15.0,  150,  37.3, "flexible SoC"),
 ("Hailo-8\n(accel+host)",         200.0,    0,       2.5,  100,  37.0, "flexible, compiled"),
 ("Coral Edge TPU\n(dev board)",   130.0,    0,       2.0,   35,  36.0, "flexible, compiled"),
 ("Desktop\nRTX 4060",             300.0,    0,     115.0,  400,  37.4, "flexible GPU"),
 ("Raspberry Pi 5\n(CPU only)",     80.0,    0,       7.0,    2,  37.3, "flexible, too slow"),
]
names=[p[0] for p in plats]; unit=np.array([p[1] for p in plats]); nre=np.array([p[2] for p in plats])
pw=np.array([p[3] for p in plats]); fps=np.array([p[4] for p in plats]); mAP=np.array([p[5] for p in plats])

fps_per_w   = fps/pw
mj_per_inf  = pw/fps*1000.0
fps_per_usd = fps/unit
capex       = nre + unit*FLEET
energy_kwh  = pw*HOURS_3YR*FLEET/1000.0
energy_cost = energy_kwh*KWH_RATE
tco         = capex + energy_cost

ASIC=0  # highlight index
hi="#1b9e77"; mut="#9aa0a6"; acc="#d95f02"
cols=[hi]+[mut]*(len(plats)-1)

# ---------- markdown tables (printed for the doc) ----------
def f(x,fmt): return format(x,fmt)
print("@@CORE@@")
print("| Platform | Type | Power | FPS @640 INT8 | mAP50-95 | Unit cost (@100k) | NRE |")
print("|---|---|---|---|---|---|---|")
typ=["Fixed-weight ASIC","Edge GPU SoC","NN accelerator","NN accelerator","Desktop GPU","CPU SBC"]
for i,p in enumerate(plats):
    ncell = ("$%.1fM"%(nre[i]/1e6)) if nre[i]>0 else "--"
    ucell = ("$%.0f"%unit[i]) if unit[i]>=10 else ("$%.0f"%unit[i])
    pwcell = ("%.0f mW"%(pw[i]*1000)) if pw[i]<1 else ("%.0f W"%pw[i])
    print("| %s | %s | %s | %d | %.1f | %s | %s |"%(p[0].replace(chr(10)," "),typ[i],pwcell,fps[i],mAP[i],ucell,ncell))

print("@@EFF@@")
print("| Platform | FPS/W (efficiency) | Energy / inference | FPS/$ |")
print("|---|---|---|---|")
for i,p in enumerate(plats):
    print("| %s | %.1f | %.1f mJ | %.3f |"%(p[0].replace(chr(10)," "),fps_per_w[i],mj_per_inf[i],fps_per_usd[i]))

print("@@TCO@@")
print("3-yr fleet TCO  (%s units, 24/7 = %d h, $%.2f/kWh):"%(format(FLEET,','),HOURS_3YR,KWH_RATE))
print("| Platform | Hardware capex | 3-yr energy | Total 3-yr TCO | vs ASIC |")
print("|---|---|---|---|---|")
for i,p in enumerate(plats):
    print("| %s | $%s | $%s | $%s | %.1fx |"%(p[0].replace(chr(10)," "),
        format(int(capex[i]),','),format(int(energy_cost[i]),','),format(int(tco[i]),','), tco[i]/tco[ASIC]))

# break-even volume vs Jetson (capex only)
N_be = nre[ASIC]/(unit[1]-unit[ASIC])
print("@@BE@@")
print("Break-even volume (hardware capex, ASIC vs Jetson Orin Nano): ~%s units"%format(int(round(N_be,-2)),','))
print("Energy/frame advantage vs Jetson: %.0fx | vs RTX system: %.0fx | vs Hailo: %.0fx"%(
    mj_per_inf[1]/mj_per_inf[ASIC], mj_per_inf[4]/mj_per_inf[ASIC], mj_per_inf[2]/mj_per_inf[ASIC]))

# ---------- chart ----------
plt.rcParams.update({"font.size":10,"axes.titlesize":12,"axes.titleweight":"bold","figure.dpi":150})
fig,ax=plt.subplots(2,2,figsize=(14,10))
fig.suptitle("Silicon YOLO  —  Taped-out fixed-weight ASIC  vs  traditional hardware (same YOLOv10n / 640 / INT8 task)",
             fontsize=14,fontweight="bold",y=0.985)
x=np.arange(len(names))

def bars(a,vals,title,ylabel,log=False,fmt="%.1f",unitlbl=""):
    b=a.bar(x,vals,color=cols,edgecolor="black",linewidth=0.5)
    a.set_title(title); a.set_ylabel(ylabel)
    a.set_xticks(x); a.set_xticklabels(names,fontsize=8)
    if log: a.set_yscale("log")
    for r,v in zip(b,vals):
        a.text(r.get_x()+r.get_width()/2, v*(1.06 if log else 1.0)+(0 if log else max(vals)*0.01),
               (fmt%v)+unitlbl, ha="center",va="bottom",fontsize=8,fontweight="bold")
    a.grid(axis="y",alpha=0.25,which="both")

bars(ax[0,0],pw,"Power draw  (lower = better)","Watts (log)",log=True,fmt="%.2f",unitlbl=" W")
bars(ax[0,1],mj_per_inf,"Energy per inference  (lower = better)","mJ / frame (log)",log=True,fmt="%.1f",unitlbl="")
bars(ax[1,0],fps_per_w,"Efficiency  (higher = better)","FPS per Watt (log)",log=True,fmt="%.1f",unitlbl="")
bars(ax[1,1],tco/1e6,"3-yr fleet TCO @100k units, 24/7  (lower = better)","Million USD (log)",log=True,fmt="%.2f",unitlbl="M")

leg=[Patch(facecolor=hi,edgecolor="k",label="Silicon YOLO ASIC (this design)"),
     Patch(facecolor=mut,edgecolor="k",label="Traditional hardware")]
fig.legend(handles=leg,loc="lower center",ncol=2,frameon=False,fontsize=10,bbox_to_anchor=(0.5,-0.005))
fig.text(0.5,0.022,"Estimates. ASIC: 28nm, ~5 mm^2 die, weights as CSD constants (0 DSP, no external DRAM); unit cost at 100k+ vol; +~2.5M USD NRE.  "
         "Same frozen INT8 weights -> ~equal mAP.  TCO at 0.15 USD/kWh, 100k units, 24/7 for 3 yr.",ha="center",fontsize=7.5,style="italic",color="#444")
plt.tight_layout(rect=[0,0.05,1,0.97])
out=r"C:\Users\light\AppData\Roaming\simular-unified-ui\SimularFiles\artifacts\silicon_yolo_asic_cost_comparison.png"
plt.savefig(out,bbox_inches="tight",facecolor="white"); print("@@SAVED@@"+out)