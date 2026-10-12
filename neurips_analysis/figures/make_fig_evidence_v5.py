"""Figure 1 of the NAACL paper: same simulation, different evidence.

Item 121 (pH after adding 0.001 mol H2SO4, key B = 2.751) and GPT-5.1 trajectories on it:
  inline                         -> report in context, answers B
  TOC (rerun 1)                  -> reads only sel.out (2.75065), answers A
  TOC + rounding hint (run 1)    -> reads only q1_step1.sel (2.75065), answers A
  TOC + report hint (run 1)      -> reads result.out lines 108-132, answers B
Right panel: share of trajectories that never receive report content, and the share answered wrongly once
the keyed value has been seen (100 - closure), GPT-5.1 / GPT-5.2 over all numeric items (paper Tables 4 and 5).

Usage: python make_fig_evidence.py <out.pdf>
Style: Times for formal labels, Excalifont (hand-drawn) for annotations, rounded light-gray frame.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
font_manager.fontManager.addfont(os.path.join(HERE, "fonts", "Excalifont-Latin.ttf"))
for f in ["Times New Roman.ttf", "Times New Roman Bold.ttf", "Courier New.ttf", "Courier New Bold.ttf"]:
    p = os.path.join("/System/Library/Fonts/Supplemental", f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Times New Roman", "mathtext.fontset": "stix"})

HAND, MONO = "Excalifont", "Courier New"
# Two hue families: cool steel blue = simulator report and correct answers,
# warm terracotta = the agent's own file, wrong answers, and failure rates.
REPORT, REPORT_BG = "#3065a0", "#e6f0fb"
OWN, OWN_BG, OWN_TEXT = "#d07546", "#feece1", "#a64627"
BAR_FILL = {"5.1": "#f3c9b0", "5.2": "#de9872"}
BAR_EDGE, TRACK = "#d07546", "#efefef"
DARK, GRAY, LINE, FRAME = "#1e1e1e", "#5e5e5e", "#858585", "#c8c8c8"
BLUE, BLUE_BG, ORANGE, ORANGE_BG = REPORT, REPORT_BG, OWN, OWN_BG
GREEN, RED = REPORT, OWN_TEXT

# --------------------------------------------------------------------- data (paper macros)
# per row: (skip report %, closure %) for GPT-5.1 and GPT-5.2
RATES = [
    ((0.0, 94.4), (0.0, 95.3)),     # inline (logged run)
    ((53.0, 85.3), (60.3, 82.4)),   # TOC (three reruns)
    ((45.9, 81.8), (55.1, 81.5)),   # TOC + rounding hint
    ((1.6, 99.2), (0.0, 98.8)),     # TOC + report hint
]

W, H = 6.3, 2.75                       # inches; drawn in units of 0.1 in
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W * 10)
ax.set_ylim(0, H * 10)
ax.axis("off")


def box(x0, y0, x1, y1, ec, fc, lw=1.0, r=0.8, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=%g" % r,
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def arrow(p0, p1, color=LINE, lw=0.8, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=3.0,head_width=1.7", color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def text(x, y, s, size=7.4, font=None, color=DARK, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, family=font or "Times New Roman", color=color, weight=weight,
            ha=ha, va=va, **kw)


def check(x, y, s=0.7, color=GREEN):
    ax.plot([x - 0.9 * s, x - 0.25 * s, x + 1.0 * s], [y + 0.05 * s, y - 0.7 * s, y + 0.95 * s],
            color=color, lw=1.3, solid_capstyle="round", solid_joinstyle="round", zorder=4)


def cross(x, y, s=0.55, color=RED):
    for dx in (-1, 1):
        ax.plot([x - s, x + s], [y - dx * s, y + dx * s], color=color, lw=1.3, solid_capstyle="round", zorder=4)


def brace(x, y0, y1, w, color=LINE, lw=0.8):
    ym = (y0 + y1) / 2
    t = np.linspace(0, 1, 60)
    for a, b in [(y0, ym), (y1, ym)]:
        yy = a + (b - a) * t
        xx = x + 0.45 * w * np.sin(np.pi * np.minimum(t, 0.5)) + 0.55 * w * np.clip((t - 0.8) / 0.2, 0, 1) ** 2
        ax.plot(xx, yy, color=color, lw=lw, solid_capstyle="round", zorder=2)


T, TB, HD, MO = 7.4, 7.7, 6.3, 7.0     # Times, Times bold, hand-drawn, mono (pt)

# ------------------------------------------------------------------ frame and column headers
box(0.4, 0.4, W * 10 - 0.4, H * 10 - 0.4, FRAME, "white", lw=0.7, r=1.6, z=0)
L0, L1 = 1.3, 17.6                     # left panel
C0, C1 = 20.2, 31.4                    # condition boxes
R0, R1 = 32.9, 43.1                    # what the agent answers from
A0, A1 = 44.5, 48.0                    # answer badge
XM = 49.6                              # model tag
S0, SS = 50.9, 0.058                   # 'skips report' bars: start, units per %
E0, ES = 57.0, 0.17                    # 'wrong once seen' bars: start, units per %
rows = [22.1, 16.2, 10.3, 4.4]
hh = 2.1
HY = 25.9
text((C0 + C1) / 2, HY, "delivery", size=HD, font=HAND, color=GRAY)
text((R0 + R1) / 2, HY, "answers from", size=HD, font=HAND, color=GRAY)
text((A0 + A1) / 2, HY, "answer", size=HD, font=HAND, color=GRAY)
text(S0 + 2.2, HY + 0.55, "skips", size=HD, font=HAND, color=GRAY)
text(S0 + 2.2, HY - 0.55, "report", size=HD, font=HAND, color=GRAY)
text(E0 + 2.2, HY + 0.55, "wrong once", size=HD, font=HAND, color=GRAY)
text(E0 + 2.2, HY - 0.55, "value seen", size=HD, font=HAND, color=GRAY)

# ------------------------------------------------------------------ left: item and the two files
box(L0, 18.0, L1, 24.8, LINE, "white", lw=0.7)
text((L0 + L1) / 2, HY, "item", size=HD, font=HAND, color=GRAY)
text(L0 + 0.8, 23.5, "Pure water + 0.001 mol H$_2$SO$_4$", size=T, ha="left")
text(L0 + 0.8, 21.6, "pH after reaction step 1?", size=T, ha="left")
for i, (lab, val) in enumerate([("A", "2.750"), ("B", "2.751"), ("C", "2.748"), ("D", "2.745")]):
    key = lab == "B"
    text(L0 + 0.8 + i * 3.85, 19.3, "%s %s" % (lab, val), size=T, ha="left", weight="bold" if key else "normal",
         color=GREEN if key else DARK)
text((L0 + L1) / 2, 16.4, "its simulation writes both files", size=HD, font=HAND, color=GRAY)
F = (L1 - L0 - 0.8) / 2
fy0, fy1 = 3.4, 14.8
box(L0, fy0, L0 + F, fy1, BLUE, BLUE_BG, lw=0.8)
text(L0 + F / 2, 13.1, "result.out", size=MO, font=MONO, weight="bold")
text(L0 + F / 2, 11.0, "simulator report", size=HD, font=HAND, color=GRAY)
text(L0 + F / 2, 9.0, "rounded to the", size=HD - 0.4, font=HAND, color=GRAY)
text(L0 + F / 2, 8.0, "options' precision", size=HD - 0.4, font=HAND, color=GRAY)
text(L0 + F / 2, 5.4, "pH = 2.751", size=MO + 0.4, font=MONO, color=BLUE, weight="bold")
box(L1 - F, fy0, L1, fy1, ORANGE, ORANGE_BG, lw=0.8)
text(L1 - F / 2, 13.1, "sel.out", size=MO, font=MONO, weight="bold")
text(L1 - F / 2, 11.0, "agent's own file", size=HD, font=HAND, color=GRAY)
text(L1 - F / 2, 9.0, "more digits than", size=HD - 0.4, font=HAND, color=GRAY)
text(L1 - F / 2, 8.0, "the options", size=HD - 0.4, font=HAND, color=GRAY)
text(L1 - F / 2, 5.4, "2.75065", size=MO + 0.4, font=MONO, color=OWN_TEXT, weight="bold")

# brace and fork
ym = (fy0 + fy1) / 2
brace(L1 + 0.25, fy0, fy1, 0.7)
fork_x = 19.0
ax.plot([L1 + 0.95, fork_x], [ym, ym], color=LINE, lw=0.8, zorder=2)
ax.plot([fork_x, fork_x], [rows[-1], rows[0]], color=LINE, lw=0.8, zorder=2)

# ------------------------------------------------------------------ rows
conds = [("Inline", "report in context"),
         ("TOC (retrieval)", "section index only"),
         ("TOC + rounding hint", "“round, don't truncate”"),
         ("TOC + report hint", "“answer from result.out”")]
reads = [("result.out, inline", "pH = 2.751", BLUE, BLUE_BG),
         ("sel.out", "2.75065 → 2.750", ORANGE, ORANGE_BG),
         ("q1_step1.sel", "2.75065 → 2.750", ORANGE, ORANGE_BG),
         ("result.out", "pH = 2.751", BLUE, BLUE_BG)]
answers = [("B", True), ("A", False), ("A", False), ("B", True)]
notes = [None, "never opens result.out", "told to round, still truncates", None]

for y, (name, sub), (rlab, rval, rc, rbg), (ans, ok), note, rate in zip(rows, conds, reads, answers, notes, RATES):
    arrow((fork_x, y), (C0 - 0.1, y))
    box(C0, y - hh, C1, y + hh, LINE, "white", lw=0.7)
    text((C0 + C1) / 2, y + 0.85, name, size=TB, weight="bold")
    text((C0 + C1) / 2, y - 1.1, sub, size=HD, font=HAND, color=GRAY)
    arrow((C1, y), (R0 - 0.1, y))
    box(R0, y - hh, R1, y + hh, rc, rbg, lw=0.8)
    text((R0 + R1) / 2, y + 0.9, rlab, size=MO - 0.4, font=MONO, color=DARK)
    text((R0 + R1) / 2, y - 1.05, rval, size=MO, font=MONO, color=REPORT if rc == REPORT else OWN_TEXT, weight="bold")
    arrow((R1, y), (A0 - 0.1, y))
    col = REPORT if ok else OWN
    box(A0, y - 1.7, A1, y + 1.7, col, "white", lw=0.8, r=0.9)
    text(A0 + 1.15, y, ans, size=8.8, weight="bold", color=DARK)
    (check if ok else cross)(A0 + 2.5, y - 0.05)
    if note:
        text((R0 + R1) / 2, y - hh - 0.85, note, size=HD - 0.2, font=HAND, color=RED)
    # right panel: two models per row; warm bars (both columns measure failure), shade by model,
    # light track = full scale of the column (0-60% and 0-20%)
    for (skip, clo), dy, tag in zip(rate, (0.85, -0.85), ("5.1", "5.2")):
        yy = y + dy
        text(XM, yy, tag, size=HD - 0.4, font=HAND, color=GRAY)
        for x0, sc, v, vmax in [(S0, SS, skip, 60.0), (E0, ES, 100 - clo, 20.0)]:
            ax.add_patch(Rectangle((x0, yy - 0.45), vmax * sc, 0.9, linewidth=0, facecolor=TRACK, zorder=1))
            if v >= 0.5:
                ax.add_patch(Rectangle((x0, yy - 0.45), max(v * sc, 0.3), 0.9, linewidth=0.6,
                                       edgecolor=BAR_EDGE, facecolor=BAR_FILL[tag], zorder=2))
            text(x0 + vmax * sc + 0.3, yy, "%d%%" % round(v), size=T - 0.6, ha="left", color=GRAY)

text(54.3, 1.5, "GPT-5.1 / GPT-5.2, all numeric items", size=HD - 0.7, font=HAND, color=GRAY)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fig_evidence.pdf")
fig.savefig(out)
fig.savefig(os.path.splitext(out)[0] + ".png", dpi=220)
print("wrote", out)
