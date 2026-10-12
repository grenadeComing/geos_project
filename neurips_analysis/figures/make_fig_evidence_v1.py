"""Figure 1 of the NAACL paper: same simulation, different evidence.

Item 121 (pH after adding 0.001 mol H2SO4, key B = 2.751) and GPT-5.2's trajectories on it:
inline -> reads the report, answers B; TOC -> reads only its own sel1.txt (2.75065), answers A;
TOC + instruction -> reads result.out, answers B. Rates on the right are GPT-5.2 / GPT-5.1 over all
numeric items (paper Tables 4 and 5).

Usage: python make_fig_evidence.py <out.pdf>
Style: Times for formal labels, Excalifont (hand-drawn) for annotations, rounded light-gray frame.
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
for f in ["Times New Roman.ttf", "Times New Roman Bold.ttf"]:
    p = os.path.join("/System/Library/Fonts/Supplemental", f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Times New Roman", "mathtext.fontset": "stix"})

HAND = "Excalifont"
MONO = "DejaVu Sans Mono"
ORANGE, BLUE, RED, GREEN = "#eb6834", "#2a78d6", "#b3261e", "#2e7d32"
DARK, GRAY, FRAME = "#1e1e1e", "#6d6d6d", "#b0b0b0"
BLUE_BG, ORANGE_BG, GRAY_BG = "#e1e9f3", "#fbe3d8", "#f3f2ef"

W, H = 6.3, 2.5                      # inches; drawn in units of 0.1 in
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W * 10)
ax.set_ylim(0, H * 10)
ax.axis("off")


def box(x0, y0, x1, y1, ec, fc, lw=1.0, r=0.8, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=%g" % r,
                                linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z))


def arrow(p0, p1, color=GRAY, lw=1.1, rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=3.2,head_width=1.8", color=color, lw=lw,
                                 connectionstyle="arc3,rad=%g" % rad, shrinkA=0, shrinkB=0, zorder=z))


def text(x, y, s, size=8.0, font=None, color=DARK, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, family=font or "Times New Roman", color=color, weight=weight,
            ha=ha, va=va, **kw)


def brace(x, y0, y1, w, color=GRAY, lw=1.0):
    """Hand-drawn curly brace along x from y0 to y1, opening to the left, tip at (x + w, mid)."""
    import numpy as np
    ym = (y0 + y1) / 2
    t = np.linspace(0, 1, 60)
    for a, b in [(y0, ym), (y1, ym)]:
        yy = a + (b - a) * t
        xx = x + w * 0.5 * (1 - np.cos(np.pi * t)) * 0.55 + w * 0.45 * np.clip((t - 0.85) / 0.15, 0, 1) ** 2
        xx = x + 0.45 * w * np.sin(np.pi * np.minimum(t, 0.5)) + 0.55 * w * np.clip((t - 0.8) / 0.2, 0, 1) ** 2
        ax.plot(xx, yy, color=color, lw=lw, solid_capstyle="round", zorder=2)


def check(x, y, s=1.0, color=GREEN):
    ax.plot([x - 0.9 * s, x - 0.25 * s, x + 1.0 * s], [y + 0.05 * s, y - 0.7 * s, y + 0.95 * s],
            color=color, lw=1.8, solid_capstyle="round", solid_joinstyle="round", zorder=4)


def cross(x, y, s=0.8, color=RED):
    for dx in (-1, 1):
        ax.plot([x - s, x + s], [y - dx * s, y + dx * s], color=color, lw=1.8, solid_capstyle="round", zorder=4)


# ------------------------------------------------------------------ sizes
T, TB, HD, MO = 7.6, 7.9, 6.4, 7.2      # Times, Times bold labels, hand-drawn, mono (pt)

# ------------------------------------------------------------------ outer frame
box(0.4, 0.4, W * 10 - 0.4, H * 10 - 0.4, FRAME, "white", lw=0.9, r=1.6, z=0)

# ------------------------------------------------------------------ left: item and the two files
L0, L1 = 1.4, 17.4
box(L0, 15.6, L1, 23.6, "#8f8f8f", GRAY_BG, lw=0.9)
text(L0 + 0.8, 22.1, "Item.", size=TB, weight="bold", ha="left")
text(L0 + 3.7, 22.1, "Pure water + 0.001 mol H$_2$SO$_4$.", size=T, ha="left")
text(L0 + 0.8, 20.1, "pH after reaction step 1?", size=T, ha="left")
for i, (lab, val) in enumerate([("A", "2.750"), ("B", "2.751"), ("C", "2.748"), ("D", "2.745")]):
    key = lab == "B"
    text(L0 + 0.8 + i * 3.85, 17.4, "%s %s" % (lab, val), size=T, ha="left", weight="bold" if key else "normal",
         color=GREEN if key else DARK)
text((L0 + L1) / 2, 13.9, "its simulation writes both files", size=HD, font=HAND, color=GRAY)
F = (L1 - L0 - 0.8) / 2
box(L0, 3.6, L0 + F, 12.2, BLUE, BLUE_BG, lw=1.1)
text(L0 + F / 2, 10.6, "result.out", size=MO, font=MONO, weight="bold")
text(L0 + F / 2, 8.6, "simulator report", size=HD, font=HAND, color=GRAY)
text(L0 + F / 2, 5.9, "pH = 2.751", size=MO + 0.4, font=MONO, color=BLUE, weight="bold")
box(L1 - F, 3.6, L1, 12.2, ORANGE, ORANGE_BG, lw=1.1)
text(L1 - F / 2, 10.6, "sel1.txt", size=MO, font=MONO, weight="bold")
text(L1 - F / 2, 8.6, "agent's own file", size=HD, font=HAND, color=GRAY)
text(L1 - F / 2, 5.9, "2.75065", size=MO + 0.4, font=MONO, color=ORANGE, weight="bold")
text(L1 - F / 2, 2.2, "more digits", size=HD, font=HAND, color=ORANGE)

# ------------------------------------------------------------------ middle: three conditions
rows = [20.4, 12.9, 5.4]
hh = 2.45                                    # half height of the row boxes
C0, C1 = 19.8, 31.0                          # condition boxes
R0, R1 = 32.6, 43.2                          # what the agent reads
A0, A1 = 44.8, 48.6                          # answer badge
conds = [("Inline", "report in context"),
         ("TOC (retrieval)", "section index only"),
         ("TOC + instruction", "\u201canswer from result.out\u201d")]
reads = [("reads the report", "pH = 2.751", BLUE, BLUE_BG),
         ("reads only sel1.txt", "2.75065 \u2192 2.750", ORANGE, ORANGE_BG),
         ("reads the report", "pH = 2.751", BLUE, BLUE_BG)]
answers = [("B", True), ("A", False), ("B", True)]

fork_x = 18.9
ym = (3.6 + 12.2) / 2
brace(L1 + 0.25, 3.6, 12.2, 0.75)              # groups the two files; tip at (L1 + 1.0, ym)
ax.plot([L1 + 1.0, fork_x], [ym, ym], color=GRAY, lw=1.1, zorder=2)
ax.plot([fork_x, fork_x], [rows[2], rows[0]], color=GRAY, lw=1.1, zorder=2)
for y, (name, sub), (rlab, rval, rc, rbg), (ans, ok) in zip(rows, conds, reads, answers):
    arrow((fork_x, y), (C0 - 0.1, y))
    box(C0, y - hh, C1, y + hh, "#8f8f8f", GRAY_BG, lw=0.9)
    text((C0 + C1) / 2, y + 0.9, name, size=TB, weight="bold")
    text((C0 + C1) / 2, y - 1.15, sub, size=HD, font=HAND, color=GRAY)
    arrow((C1, y), (R0 - 0.1, y))
    box(R0, y - hh, R1, y + hh, rc, rbg, lw=1.1)
    text((R0 + R1) / 2, y + 0.95, rlab, size=HD, font=HAND, color=DARK)
    text((R0 + R1) / 2, y - 1.1, rval, size=MO, font=MONO, color=rc, weight="bold")
    arrow((R1, y), (A0 - 0.1, y))
    col = GREEN if ok else RED
    box(A0, y - 1.9, A1, y + 1.9, col, "#e8f3e9" if ok else "#f8e3e1", lw=1.1, r=0.9)
    text(A0 + 1.25, y, ans, size=9.0, weight="bold", color=DARK)
    (check if ok else cross)(A0 + 2.75, y - 0.05, 0.7 if ok else 0.55)

# annotation on the failure row
text((R0 + R1) / 2, 9.55, "never opens result.out,", size=HD - 0.2, font=HAND, color=RED)
text((R0 + R1) / 2, 8.55, "truncates instead of rounding", size=HD - 0.2, font=HAND, color=RED)

# ------------------------------------------------------------------ right: rates, GPT-5.2 / GPT-5.1
xs, xc = 52.6, 58.6
text(xs, 23.4, "skips report", size=HD + 0.2, font=HAND, color=DARK)
text(xc, 23.4, "closure", size=HD + 0.2, font=HAND, color=DARK)
ax.plot([49.9, 61.6], [22.4, 22.4], color=FRAME, lw=0.6)
rates = [("0% / 0%", "95% / 94%", DARK), ("60% / 53%", "82% / 85%", RED), ("0% / 2%", "99% / 99%", DARK)]
for y, (sk, cl, c) in zip(rows, rates):
    text(xs, y, sk, size=T, color=c)
    text(xc, y, cl, size=T, color=c)
text((xs + xc) / 2, 2.2, "GPT-5.2 / GPT-5.1", size=HD, font=HAND, color=GRAY)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fig_evidence.pdf")
fig.savefig(out)
fig.savefig(os.path.splitext(out)[0] + ".png", dpi=220)
print("wrote", out)
