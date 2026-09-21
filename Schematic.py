#!/usr/bin/env python3
"""
Two black-and-white schematics for §4 of the ENGR 839 test plan.

  ps_setup_schematic.png            Fig. 4.1  hybrid-simulation test setup
  ps_instrumentation_schematic.png  Fig. 4.2  sensor layout on the PS specimen

Dimensions in the drawings are model scale (mm) for the specimen; the numerical
substructure is full scale. Pure line art, no colour, so it prints cleanly.

    python ps_schematics.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "hatch.linewidth": 0.5})
K = "black"

# ---- specimen geometry (mm) ------------------------------------------------
SEG = 433.3; NSEG = 6; H_PS = SEG * NSEG          # 2600
W = 1000                                          # width in the plane of the bridge
FOOT_W, FOOT_H = 2600, 700
TOP_W, TOP_H = 1200, 500


def specimen(ax, x0=0.0, hatch_footing=True, show_tendon=True, show_ed=True, lw=1.0):
    """Draw footing, six segments, top block. x0 = left edge of the specimen."""
    # strong floor
    ax.plot([x0 - 1500, x0 + W + 1500], [-FOOT_H, -FOOT_H], color=K, lw=1.6)
    for xx in np.arange(x0 - 1500, x0 + W + 1500, 120):
        ax.plot([xx, xx - 80], [-FOOT_H, -FOOT_H - 80], color=K, lw=0.6)
    # footing
    fx = x0 + W / 2 - FOOT_W / 2
    ax.add_patch(Rectangle((fx, -FOOT_H), FOOT_W, FOOT_H, fc="white", ec=K, lw=lw,
                           hatch="//" if hatch_footing else None))
    # segments
    for i in range(NSEG):
        ax.add_patch(Rectangle((x0, i * SEG), W, SEG, fc="white", ec=K, lw=lw))
        # hollow core shown dashed
        ax.add_patch(Rectangle((x0 + 75, i * SEG + 4), W - 150, SEG - 8, fc="none", ec=K, lw=0.5, ls=(0, (2, 2))))
    # joint ticks and labels
    for j in range(NSEG + 1):
        y = j * SEG
        ax.plot([x0 - 60, x0], [y, y], color=K, lw=0.8)
        ax.text(x0 - 75, y, f"J{j+1}", ha="right", va="center", fontsize=7.5)
    # top block
    ax.add_patch(Rectangle((x0 + W / 2 - TOP_W / 2, H_PS), TOP_W, TOP_H, fc="white", ec=K, lw=lw, hatch="//"))
    if show_tendon:
        xc = x0 + 0.4565 * W                       # centroid at 5.478/12
        ax.plot([xc, xc], [-FOOT_H + 120, H_PS + TOP_H - 60], color=K, lw=1.2, ls=(0, (6, 3)))
        ax.plot([xc - 70, xc + 70], [-FOOT_H + 120] * 2, color=K, lw=2)      # dead-end anchor
        ax.plot([xc - 70, xc + 70], [H_PS + TOP_H - 60] * 2, color=K, lw=2)  # live-end anchor
    if show_ed:
        for xe in (x0 + 100, x0 + 200, x0 + W - 200, x0 + W - 100):
            ax.plot([xe, xe], [-350, 300], color=K, lw=1.6)
    return fx


def arrow(ax, p, q, text=None, lw=1.2, style="-|>", ms=10, tpos=None, **kw):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=K, **kw)
    ax.add_patch(a)
    if text:
        tx, ty = tpos if tpos else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(tx, ty, text, ha="center", va="center", fontsize=8,
                bbox=dict(fc="white", ec="none", pad=1))


def actuator(ax, p, q, label, lw=1.4):
    """Hydraulic actuator drawn between hinge p and hinge q."""
    (x0, y0), (x1, y1) = p, q
    ax.plot([x0, x1], [y0, y1], color=K, lw=lw)
    # cylinder body: thick segment in the middle 45 %
    t0, t1 = 0.30, 0.75
    ax.plot([x0 + t0 * (x1 - x0), x0 + t1 * (x1 - x0)], [y0 + t0 * (y1 - y0), y0 + t1 * (y1 - y0)], color=K, lw=lw * 5, solid_capstyle="butt")
    for (x, y) in (p, q):
        ax.add_patch(Circle((x, y), 28, fc="white", ec=K, lw=1))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=7.5,
            bbox=dict(fc="white", ec=K, lw=0.6, pad=1.5))


# ============================================================================
# Fig. 4.1  test setup
# ============================================================================
fig = plt.figure(figsize=(12.0, 6.4))
ax = fig.add_axes([0.01, 0.05, 0.64, 0.90])
ax.set_aspect("equal"); ax.axis("off")

fx = specimen(ax)

# L-beam on top block: horizontal arm to the left, vertical stub
LB_Y = H_PS + TOP_H
ax.add_patch(Rectangle((-900, LB_Y), W + 1200, 180, fc="white", ec=K, lw=1.2))
ax.add_patch(Rectangle((-900, LB_Y - 420), 180, 600, fc="white", ec=K, lw=1.2))
ax.text(-300, LB_Y - 520, "L-shaped\nloading beam", ha="center", va="top", fontsize=7.5)

# reaction wall on the left
ax.add_patch(Rectangle((-2350, -FOOT_H), 300, 4400, fc="white", ec=K, lw=1.4, hatch="\\\\"))
ax.text(-2200, 3800, "reaction\nwall", ha="center", va="center", fontsize=7.5, rotation=90)
# horizontal actuator
actuator(ax, (-2050, LB_Y - 120), (-900, LB_Y - 120), "Actuator 1\nhorizontal\n500 kN")

# portal reaction frame above
PF_Y = LB_Y + 1500
ax.add_patch(Rectangle((-1300, PF_Y), 3500, 200, fc="white", ec=K, lw=1.4))
for xx in (-1300, 2000):
    ax.add_patch(Rectangle((xx, -FOOT_H), 200, PF_Y + FOOT_H + 200, fc="white", ec=K, lw=1.2))
ax.text(450, PF_Y + 320, "portal reaction frame (two frames, one behind the other, also restrain out-of-plane motion)",
        ha="center", va="center", fontsize=7.5)
# vertical actuators
for xx, lab in ((-250, "Actuator 2\nvertical\n1500 kN"), (1250, "Actuator 3\nvertical\n1500 kN")):
    actuator(ax, (xx, PF_Y), (xx, LB_Y + 180), lab)
ax.annotate("", xy=(-250, LB_Y + 400), xytext=(1250, LB_Y + 400), arrowprops=dict(arrowstyle="<->", lw=0.8, color=K))
ax.text(500, LB_Y + 470, "1500 mm", ha="center", fontsize=7.5)

# interface DOFs at top of PS
xi, yi = W / 2, H_PS
arrow(ax, (xi + 700, yi + 250), (xi + 1100, yi + 250), "u", tpos=(xi + 1180, yi + 250))
arrow(ax, (xi + 700, yi + 250), (xi + 700, yi + 650), "v", tpos=(xi + 700, yi + 730))
th = np.linspace(-0.6, 2.2, 40); r = 260
ax.plot(xi + 700 + r * np.cos(th), yi + 250 + r * np.sin(th), color=K, lw=1.0)
ax.add_patch(FancyArrowPatch((xi + 700 + r * np.cos(th[-3]), yi + 250 + r * np.sin(th[-3])),
                             (xi + 700 + r * np.cos(th[-1]), yi + 250 + r * np.sin(th[-1])), arrowstyle="-|>", mutation_scale=9, color=K))
ax.text(xi + 1000, yi + 560, "θ", fontsize=9)

# labels: annotation column to the right of the frame, with leader lines
AX = 2450
def note(ax, text, target, y, ha="left"):
    ax.plot([target[0], AX - 60], [target[1], y], color=K, lw=0.6)
    ax.text(AX, y, text, ha=ha, va="center", fontsize=7.5)
note(ax, "loading block, strand live ends,\ntendon load cells", (W / 2 + TOP_W / 2, H_PS + TOP_H / 2), H_PS + TOP_H / 2 + 500)
note(ax, "interface = top of PS: u, v, θ\n(3 DOF in the bridge plane)", (xi + 1100, yi + 250), yi + 250)
note(ax, "six precast segments, 433 mm each, 2600 mm\nR-section 1000 × 892/558, 75 mm walls\ndry joints J2–J7 with shear keys", (W, 1400), 1500)
note(ax, "2 × 15.2 mm strands, unbonded,\non the section centroid", (W * 0.4565 + 20, 800), 800)
note(ax, "8 × 20 mm ED bars across J1,\n160 mm unbonded", (W - 100, 150), 150)
note(ax, "footing, post-tensioned\nto the strong floor", (W / 2 + FOOT_W / 2, -FOOT_H / 2), -FOOT_H / 2 - 150)
ax.text(W / 2, -1000, "Specimen: PS at 1:12 (physical substructure)", ha="center", va="center", fontsize=9, fontweight="bold")

ax.set_xlim(-2600, 5400); ax.set_ylim(-1200, PF_Y + 700)

# ---- right panel: hybrid loop ------------------------------------------------
ax2 = fig.add_axes([0.66, 0.05, 0.33, 0.90]); ax2.axis("off"); ax2.set_xlim(0, 10); ax2.set_ylim(0, 11)

def box(ax, x, y, w, h, title, body="", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=K, lw=lw))
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=7.8, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.25, body, ha="center", va="center", fontsize=7.5, linespacing=1.3)

box(ax2, 0.3, 7.6, 9.4, 2.3, "COORDINATOR",
    "M ü(t) + C u̇(t) + R_n(t) + R_e(t) = F(t)\nexplicit Newmark, Δt = 0.005 s, slow (non-real-time)")
box(ax2, 0.3, 3.6, 4.55, 3.3, "NUMERICAL SUBSTRUCTURE",
    "OpenSees, full scale\n95 m of pier above PS,\nsecond pier, girder, side spans\nrocking joints every 5.2 m\nRayleigh 2 % on modes 1 & 3")
box(ax2, 5.15, 3.6, 4.55, 3.3, "PHYSICAL SUBSTRUCTURE",
    "PS at 1:12\nMTS controller,\nouter-loop displacement\ncontrol on 5 LVDTs\n3 actuators")
box(ax2, 0.3, 0.4, 9.4, 2.3, "NUMERICAL TWIN OF PS  (O4)",
    "same joint elements, tendon length as built;\ncalibrated in Phase A, used for sensitivity runs")

# arrows coordinator <-> substructures
arrow(ax2, (2.0, 7.6), (2.0, 6.9), "u (full scale)", tpos=(1.3, 7.25))
arrow(ax2, (3.2, 6.9), (3.2, 7.6), "R_n", tpos=(3.75, 7.25))
arrow(ax2, (6.7, 7.6), (6.7, 6.9), "u / 12, θ × 1", tpos=(6.05, 7.25))
arrow(ax2, (8.7, 6.9), (8.7, 7.6), "F × 144\nM × 1728", tpos=(9.35, 7.25))
ax2.plot([7.4, 7.4], [2.7, 3.6], color=K, lw=0.8)
ax2.text(7.4, 3.15, "compare", ha="center", va="center", fontsize=7.5, bbox=dict(fc="white", ec="none", pad=1))
ax2.text(5.0, 10.8, "Fig. 4.1  Hybrid-simulation test setup", ha="center", va="top", fontsize=9.5, fontweight="bold")

fig.savefig("ps_setup_schematic.png", dpi=220, facecolor="white", bbox_inches="tight")
fig.savefig("ps_setup_schematic.pdf", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ============================================================================
# Fig. 4.2  instrumentation
# ============================================================================
fig = plt.figure(figsize=(9.0, 6.6))
ax = fig.add_axes([0.02, 0.04, 0.66, 0.92]); ax.set_aspect("equal"); ax.axis("off")
fx = specimen(ax, show_tendon=True, show_ed=True)


def lvdt(ax, x, y, horizontal=False, size=90, label=None):
    """LVDT symbol: small rectangle with a probe line."""
    if horizontal:
        ax.add_patch(Rectangle((x - size, y - 22), size, 44, fc="white", ec=K, lw=0.9))
        ax.plot([x, x + size * 0.7], [y, y], color=K, lw=0.9)
    else:
        ax.add_patch(Rectangle((x - 22, y - size / 2), 44, size, fc="white", ec=K, lw=0.9))
        ax.plot([x, x], [y + size / 2, y + size * 1.1], color=K, lw=0.9)
    if label:
        ax.text(x + (size * 0.8 if horizontal else 40), y, label, fontsize=6.5, va="center")


def gauge(ax, x, y):
    ax.add_patch(Rectangle((x - 25, y - 12), 50, 24, fc=K, ec=K, lw=0))


# 1 joint-opening LVDTs, both faces, J1..J7 (vertical, straddling the joint)
for j in range(NSEG + 1):
    y = j * SEG
    lvdt(ax, -55, y, size=100)
    lvdt(ax, W + 55, y, size=100)
# 2 sliding LVDTs (horizontal) on the right face just above each joint
for j in range(NSEG + 1):
    lvdt(ax, W + 150, j * SEG + 110, horizontal=True, size=70)
# 3 lateral profile LVDTs from an independent reference frame on the left
RF_X = -700
ax.add_patch(Rectangle((RF_X - 40, -FOOT_H), 40, H_PS + TOP_H + 300, fc="white", ec=K, lw=1.0, hatch="xx"))
ax.text(RF_X - 120, H_PS / 2, "independent reference frame", rotation=90, ha="center", va="center", fontsize=7.5)
for i in range(1, NSEG + 1):
    y = i * SEG - 30
    ax.plot([RF_X, -160], [y, y], color=K, lw=0.7)
    lvdt(ax, -160, y, horizontal=True, size=70)
# 4 interface LVDTs: footing lateral, top lateral, three vertical on the loading beam
lvdt(ax, -160, -FOOT_H / 2, horizontal=True, size=70)
ax.plot([RF_X, -230], [-FOOT_H / 2] * 2, color=K, lw=0.7)
LB_Y = H_PS + TOP_H
ax.add_patch(Rectangle((-500, LB_Y), W + 800, 160, fc="white", ec=K, lw=1.0))
lvdt(ax, -160, LB_Y + 80, horizontal=True, size=70); ax.plot([RF_X, -230], [LB_Y + 80] * 2, color=K, lw=0.7)
for xx in (-350, W / 2, W + 200):
    lvdt(ax, xx, LB_Y + 320, size=100)
    ax.plot([xx, xx], [LB_Y + 160, LB_Y + 270], color=K, lw=0.7)
# 5 tendon load cells at the live end (top block)
xc = 0.4565 * W
ax.add_patch(Rectangle((xc - 90, H_PS + TOP_H - 130), 180, 60, fc="white", ec=K, lw=1.0))
ax.text(xc + 110, H_PS + TOP_H - 100, "hollow load cell\n(one per strand)", fontsize=6.5, va="center")
# 6 ED bar strain gauges on the unbonded length across J1
for xe in (100, 200, W - 200, W - 100):
    gauge(ax, xe, 40); gauge(ax, xe, -60)
# 7 toe strain gauges on corner angles, both ends of segment 1
for xe in (0, W):
    for ye in (30, SEG - 30):
        ax.add_patch(Rectangle((xe - 18, ye - 50), 36, 100, fc="white", ec=K, lw=1.2))
        gauge(ax, xe, ye)
# 8 diagonal LVDT rosettes on segments 1 and 2 (webs are the faces in this elevation)
for i in range(2):
    y0, y1 = i * SEG + 60, (i + 1) * SEG - 60
    ax.plot([140, W - 140], [y0, y1], color=K, lw=0.9, ls=(0, (4, 2)))
    ax.plot([140, W - 140], [y1, y0], color=K, lw=0.9, ls=(0, (4, 2)))
    for (px, py) in ((140, y0), (W - 140, y1), (140, y1), (W - 140, y0)):
        ax.add_patch(Circle((px, py), 16, fc="white", ec=K, lw=0.9))
# 9 DIC field on segments 1-3 (north face = this face)
ax.add_patch(Rectangle((60, 20), W - 120, 3 * SEG - 40, fc="none", ec=K, lw=1.0, ls=(0, (1, 2))))
ax.text(W / 2, 3 * SEG + 40, "DIC field (segments 1–3, north face)", ha="center", va="bottom", fontsize=6.5)
# actuator load cells (symbolic) at the loading beam
for (px, py, lab) in ((-500, LB_Y + 80, "LC-H"), (-250, LB_Y + 160, "LC-V2"), (W + 250, LB_Y + 160, "LC-V3")):
    ax.add_patch(Circle((px, py), 40, fc="white", ec=K, lw=1.2)); ax.text(px, py, "LC", ha="center", va="center", fontsize=5.5)

ax.set_xlim(-1000, W + 700); ax.set_ylim(-1000, LB_Y + 600)
ax.text(W / 2, LB_Y + 560, "Fig. 4.2  Instrumentation layout on the PS specimen (elevation, bridge longitudinal direction)",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# legend panel
ax3 = fig.add_axes([0.69, 0.06, 0.30, 0.88]); ax3.axis("off"); ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)
items = [
    ("joint-opening LVDT, both faces, J1–J7 (14)", "vertical LVDT symbol at each joint"),
    ("joint-sliding LVDT, J1–J7 (7)", "horizontal LVDT symbol above each joint"),
    ("lateral profile LVDT, top of each segment (6)", "from the independent reference frame"),
    ("interface LVDTs (5)", "footing lateral, top lateral, three vertical on the loading beam"),
    ("tendon hollow load cells (2)", "live-end anchors in the top block"),
    ("ED-bar strain gauges (16)", "black rectangles on the unbonded length across J1"),
    ("toe strain gauges on corner angles (16)", "both ends of segment 1, four corners"),
    ("diagonal LVDT rosettes (8)", "dashed diagonals, segments 1 and 2"),
    ("DIC field", "dotted outline, segments 1–3"),
    ("actuator load cells (3)", "LC at the loading beam"),
]
y = 9.6
ax3.text(0, y, "Legend", fontsize=9, fontweight="bold"); y -= 0.55
for t, s in items:
    ax3.text(0, y, "• " + t, fontsize=7.5, va="top"); y -= 0.42
    ax3.text(0.35, y, s, fontsize=6.8, va="top", style="italic"); y -= 0.5
ax3.text(0, 0.35, "All channels at 50 Hz on the laboratory time axis;\nsee Table 4.2.", fontsize=7, va="bottom")

fig.savefig("ps_instrumentation_schematic.png", dpi=220, facecolor="white", bbox_inches="tight")
fig.savefig("ps_instrumentation_schematic.pdf", facecolor="white", bbox_inches="tight")
print("wrote ps_setup_schematic.png/.pdf and ps_instrumentation_schematic.png/.pdf")