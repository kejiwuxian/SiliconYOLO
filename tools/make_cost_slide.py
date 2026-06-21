
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG="#0a0e1a"; CARD="#121a2e"; TEAL="#2dd4bf"; GREEN="#34d399"; SUB="#9fb0c9"; WHITE="#f5f8ff"
fig=plt.figure(figsize=(19.2,10.8),dpi=100); fig.patch.set_facecolor(BG)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,19.2); ax.set_ylim(0,10.8); ax.axis("off")
ax.text(1.0,9.7,"SILICON YOLO   -   COST & EFFICIENCY",color=TEAL,fontsize=22,fontweight="bold")
ax.text(1.0,8.5,"Weights as silicon - the economics",color=WHITE,fontsize=58,fontweight="bold")
ax.text(1.0,7.65,"Taped-out 28nm fixed-weight ASIC  vs.  traditional hardware  -  same YOLOv10n / 640 / INT8 task",
        color=SUB,fontsize=24)
cards=[("~0.2 W","power draw","vs 15 W Jetson Orin Nano"),
       ("~\$2","per chip @ 100k vol","vs \$249 Jetson    +~\$2.5M NRE"),
       ("26x","less energy / frame","73x vs desktop GPU, 6x vs Hailo-8"),
       ("~11x","lower 3-yr fleet TCO","\$2.78M vs \$30.8M  @100k, 24/7")]
x0=1.0; w=4.2; gap=0.43; y=2.6; h=4.2
for i,(big,lab,sub) in enumerate(cards):
    x=x0+i*(w+gap)
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.25",fc=CARD,ec=TEAL,lw=2.0))
    ax.text(x+w/2,y+h-1.5,big,color=GREEN,fontsize=54,fontweight="bold",ha="center")
    ax.text(x+w/2,y+h-2.5,lab,color=WHITE,fontsize=22,fontweight="bold",ha="center")
    ax.text(x+w/2,y+0.95,sub,color=SUB,fontsize=15,ha="center",va="center")
ax.add_patch(FancyBboxPatch((1.0,1.05),17.2,0.95,boxstyle="round,pad=0.01,rounding_size=0.2",fc="#0f1626",ec="none"))
ax.text(9.6,1.52,"255 FPS/W    -    0 DSP    -    no weight SRAM / no DRAM    -    ~37.6 mAP    -    deterministic ~20 ms latency",
        color=TEAL,fontsize=16,va="center",ha="center",fontweight="bold")
ax.text(18.2,0.5,"engineering estimates - see docs/COST_COMPARISON.md",color="#6b7a94",fontsize=13,ha="right")
out=r"C:\Users\light\AppData\Roaming\simular-unified-ui\SimularFiles\artifacts\_cost_slide.png"
plt.savefig(out,facecolor=BG); print("SLIDE_SAVED "+out)
