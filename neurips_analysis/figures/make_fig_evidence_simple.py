"""Figure 1 of the NAACL paper, simplified: one item, one simulation, two files, two answers.

Item 121 (pH after adding 0.001 mol H2SO4, key B = 2.751), GPT-5.1 trajectories:
  inline          -> report in context, answers B (2.751 as printed)
  TOC retrieval   -> reads only its own sel.out (2.75065), truncates to 2.750, answers A
Palette: variant A of make_fig_evidence_restyled.py (borderless tinted cards, azure + coral).

Usage: python make_fig_evidence_simple.py <out.pdf>      (also writes <out>.png)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
font_manager.fontManager.addfont(os.path.join(HERE, "fonts", "Excalifont-Latin.ttf"))
for f in ["Times New Roman.ttf", "Times New Roman Bold.ttf", "Courier New.ttf", "Courier New Bold.ttf"]:
    p = os.path.join("/System/Library/Fonts/Supplemental", f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Times New Roman", "mathtext.fontset": "stix"})
HAND, MONO = "Excalifont", "Courier New"

# palette (variant A)
INK, SEC, HDR, LINE, FRAME = "#232933", "#5f6a77", "#78818c", "#b2b8bf", "#dadee3"
NEU = "#f4f6f9"
REP = dict(acc="#1570d1", txt="#1467c2", cont="#e2f0ff", sec="#45668a")
OWN = dict(acc="#dd5139", txt="#c63a2a", cont="#ffebe5", sec="#944a49")

W, H = 6.3, 2.05
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W * 10)
ax.set_ylim(0, H * 10)
ax.axis("off")


def card(x0, y0, x1, y1, fc, r=1.0, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=%g" % r,
                                linewidth=0, facecolor=fc, zorder=z))


def arrow(p0, p1, color=LINE, lw=0.8, rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=2.8,head_width=1.6", color=color, lw=lw,
                                 connectionstyle="arc3,rad=%g" % rad, shrinkA=0, shrinkB=0, zorder=z))


def text(x, y, s, size=7.6, font=None, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, family=font or "Times New Roman", color=color, weight=weight, ha=ha, va=va)


def check(x, y, s=0.75, color=REP["acc"]):
    ax.plot([x - 0.9 * s, x - 0.25 * s, x + 1.0 * s], [y + 0.05 * s, y - 0.7 * s, y + 0.95 * s],
            color=color, lw=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=4)


def cross(x, y, s=0.6, color=OWN["acc"]):
    for d in (-1, 1):
        ax.plot([x - s, x + s], [y - d * s, y + d * s], color=color, lw=1.5, solid_capstyle="round", zorder=4)


HD, MO = 6.6, 7.2
ax.add_patch(FancyBboxPatch((0.4, 0.4), W * 10 - 0.8, H * 10 - 0.8, boxstyle="round,pad=0,rounding_size=1.8",
                            linewidth=0.6, edgecolor=FRAME, facecolor="white", zorder=0))

YT, YB = 14.6, 5.6                       # the two paths (report above, own file below)

# ------------------------------------------------------------------ item
card(1.6, 5.4, 16.6, 14.8, NEU)
text(9.1, 19.0, "one item", size=HD, font=HAND, color=HDR)
text(2.4, 12.6, "Pure water + 0.001 mol H$_2$SO$_4$", ha="left")
text(2.4, 10.6, "pH after reaction step 1?", ha="left")
for i, (lab, val) in enumerate([("A", "2.750"), ("B", "2.751"), ("C", "2.748"), ("D", "2.745")]):
    key = lab == "B"
    text(2.4 + (i % 2) * 6.6, 8.4 - (i // 2) * 1.9, "%s  %s" % (lab, val), ha="left",
         weight="bold" if key else "normal", color=REP["txt"] if key else INK)

# ------------------------------------------------------------------ simulation forks into two files
FX0, FX1 = 21.8, 35.4
ax.plot([16.6, 19.2], [10.1, 10.1], color=LINE, lw=0.8, zorder=2)
ax.plot([19.2, 19.2], [YB, YT], color=LINE, lw=0.8, zorder=2)
arrow((19.2, YT), (FX0 - 0.1, YT))
arrow((19.2, YB), (FX0 - 0.1, YB))
text((FX0 + FX1) / 2, 19.0, "one simulation, two files", size=HD, font=HAND, color=HDR)

card(FX0, YT - 3.4, FX1, YT + 3.4, REP["cont"])
text((FX0 + FX1) / 2, YT + 2.0, "result.out", size=MO, font=MONO, weight="bold")
text((FX0 + FX1) / 2, YT + 0.1, "simulator report", size=HD - 0.2, font=HAND, color=REP["sec"])
text((FX0 + FX1) / 2, YT - 2.0, "pH = 2.751", size=MO + 0.3, font=MONO, weight="bold", color=REP["txt"])

card(FX0, YB - 3.4, FX1, YB + 3.4, OWN["cont"])
text((FX0 + FX1) / 2, YB + 2.0, "sel.out", size=MO, font=MONO, weight="bold")
text((FX0 + FX1) / 2, YB + 0.1, "agent's own file", size=HD - 0.2, font=HAND, color=OWN["sec"])
text((FX0 + FX1) / 2, YB - 2.0, "2.75065", size=MO + 0.3, font=MONO, weight="bold", color=OWN["txt"])

# ------------------------------------------------------------------ two paths to two answers
AX0, AX1 = 51.6, 60.6
arrow((FX1, YT), (AX0 - 0.1, YT), color=REP["acc"], lw=0.9)
text((FX1 + AX0) / 2, YT + 1.25, "inline: report in context", size=HD, font=HAND, color=REP["sec"])
text((FX1 + AX0) / 2, YT - 1.25, "reads 2.751 as printed", size=HD - 0.3, font=HAND, color=REP["sec"])

arrow((FX1, YB), (AX0 - 0.1, YB), color=OWN["acc"], lw=0.9)
text((FX1 + AX0) / 2, YB + 1.25, "retrieval: section index only", size=HD, font=HAND, color=OWN["sec"])
text((FX1 + AX0) / 2, YB - 1.25, "truncates 2.75065 to 2.750", size=HD - 0.3, font=HAND, color=OWN["sec"])

card(AX0, YT - 2.3, AX1, YT + 2.3, REP["cont"])
text(AX0 + 1.4, YT, "B", size=9.2, weight="bold", ha="left")
text(AX0 + 3.1, YT, "2.751", size=7.8, ha="left", color=REP["txt"])
check(AX1 - 1.4, YT - 0.05)

card(AX0, YB - 2.3, AX1, YB + 2.3, OWN["cont"])
text(AX0 + 1.4, YB, "A", size=9.2, weight="bold", ha="left")
text(AX0 + 3.1, YB, "2.750", size=7.8, ha="left", color=OWN["txt"])
cross(AX1 - 1.4, YB)
text((AX0 + AX1) / 2, 19.0, "answer", size=HD, font=HAND, color=HDR)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fig_evidence_simple.pdf")
fig.savefig(out)
fig.savefig(os.path.splitext(out)[0] + ".png", dpi=220)
print("wrote", out)
