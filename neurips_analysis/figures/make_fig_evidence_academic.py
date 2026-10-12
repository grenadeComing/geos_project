"""Figure 1 of the NAACL paper: same simulation, different evidence (academic style).

Item 121 (pH after adding 0.001 mol H2SO4, key B = 2.751) and GPT-5.1 trajectories on it:
  inline (July logged run)        -> report in context, answers B
  TOC (rerun 1)                   -> reads only sel.out (2.75065), answers A
  TOC + rounding hint (run 1)     -> reads only q1_step1.sel (2.75065), answers A
  TOC + report hint (run 1)       -> reads result.out lines 108-132, answers B
File excerpts are verbatim lines from the TOC rerun-1 workspace (result.out lines 118-121, sel.out),
with leading whitespace and columns before `step` omitted.
Panel (c): skips report / wrong after seeing the keyed value (100 - closure), GPT-5.1 filled, GPT-5.2 open,
over all numeric items (paper Tables 4 and 5).

Style (from a survey of ACL/NeurIPS figures): Times and Courier, thin gray frames with the label in the
top border, colour only on highlighted values and on the check/cross glyphs, booktabs rules, axes on plots.

Usage: python make_fig_evidence.py <out.pdf>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

SUPP = "/System/Library/Fonts/Supplemental"
for f in ["Times New Roman.ttf", "Times New Roman Bold.ttf", "Times New Roman Italic.ttf",
          "Courier New.ttf", "Courier New Bold.ttf"]:
    if os.path.exists(os.path.join(SUPP, f)):
        font_manager.fontManager.addfont(os.path.join(SUPP, f))
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Times New Roman", "mathtext.fontset": "stix"})

SERIF, MONO, GLYPH = "Times New Roman", "Courier New", "DejaVu Sans"
BLACK, DGRAY, MGRAY, LGRAY = "#000000", "#595959", "#808080", "#e0e0e0"
RED, GREEN = "#cc0000", "#38761d"
HL_REPORT, HL_OWN = "#cfe2f3", "#fff2cc"

# rates per row: (skip %, wrong-after-seeing %) for GPT-5.1, GPT-5.2
RATES = [((0.0, 5.6), (0.0, 4.7)),
         ((53.0, 14.7), (60.3, 17.6)),
         ((45.9, 18.2), (55.1, 18.5)),
         ((1.6, 0.8), (0.0, 1.2))]

Wpt, Hpt = 453.6, 194.0
fig = plt.figure(figsize=(Wpt / 72, Hpt / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, Wpt)
ax.set_ylim(0, Hpt)
ax.axis("off")


def t(x, y, s, size=8, font=SERIF, color=BLACK, weight="normal", style="normal", ha="left", va="baseline", hl=None, **kw):
    if hl:
        kw["bbox"] = dict(boxstyle="square,pad=0.08", fc=hl, ec="none")
    return ax.text(x, y, s, fontsize=size, family=font, color=color, weight=weight, style=style, ha=ha, va=va, **kw)


def hline(x0, x1, y, lw, color=BLACK):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")


def framed(x0, y0, x1, y1, label):
    """Thin gray frame with the label set into the top border."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=MGRAY, lw=0.5))
    t(x0 + 5, y1, " " + label + " ", size=7, color=DGRAY, va="center",
      bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))


# ------------------------------------------------------------------ (a) item and the two files
AX0, AX1 = 1.0, 141.0
t(AX0, 184, "(a)", size=8.5, weight="bold")
t(AX0 + 14, 184, "Item 121", size=8.5, weight="bold")
t(AX0, 171, "Pure water + 0.001 mol H$_2$SO$_4$.", size=8)
t(AX0, 161, "What is the pH after reaction step 1?", size=8)
t(AX0, 150, "(A) 2.750", size=8)
t(AX0 + 50, 150, "(B) 2.751", size=8, weight="bold")
t(AX0, 140, "(C) 2.748", size=8)
t(AX0 + 50, 140, "(D) 2.745", size=8)

framed(AX0, 82, AX1, 124, "result.out  (simulator report)")
ln = dict(size=6.6, font=MONO, color=MGRAY)
t(AX0 + 4, 111, "118", **ln)
t(AX0 + 20, 111, "---Description of solution---", size=6.6, font=MONO)
t(AX0 + 4, 100, "120", **ln)
t(AX0 + 20, 100, "pH  =", size=6.6, font=MONO)
t(AX0 + 47, 100, "2.751", size=6.6, font=MONO, hl=HL_REPORT)
t(AX0 + 72, 100, "Charge balance", size=6.6, font=MONO)
t(AX0 + 4, 89, "121", **ln)
t(AX0 + 20, 89, "pe  = 14.692   Adjusted to ...", size=6.6, font=MONO)

framed(AX0, 18, AX1, 66, "sel.out  (defined by the agent)")
t(AX0 + 4, 53, "...", size=6.6, font=MONO, color=MGRAY)
t(AX0 + 22, 53, "step        pH        pe", size=6.6, font=MONO)
t(AX0 + 4, 42, "...", size=6.6, font=MONO, color=MGRAY)
t(AX0 + 22, 42, " -99   6.99738         4", size=6.6, font=MONO)
t(AX0 + 4, 31, "...", size=6.6, font=MONO, color=MGRAY)
t(AX0 + 22, 31, "   1", size=6.6, font=MONO)
t(AX0 + 52, 31, "2.75065", size=6.6, font=MONO, hl=HL_OWN)
t(AX0 + 87, 31, "14.6924", size=6.6, font=MONO)
t(AX0, 6, "Both files come from the same input.", size=7, color=DGRAY, style="italic")

# ------------------------------------------------------------------ (b) table of conditions
BX0 = 150.0
CX = [BX0 + 2, BX0 + 82, BX0 + 140]       # condition | agent reads | answer
DX0, DX1 = 332.0, 452.0                   # dot-plot panel (c)
TOP, MID, BOT = 190.0, 170.0, 8.0
hline(BX0, DX1, TOP, 0.8)
hline(BX0, DX1, MID, 0.4)
hline(BX0, DX1, BOT, 0.8)
t(BX0 - 0.5, 178, "(b)", size=8.5, weight="bold")
t(CX[0] + 13, 178, "Condition", size=8, weight="bold")
t(CX[1], 178, "Agent reads", size=8, weight="bold")
t(CX[2], 178, "Answer", size=8, weight="bold")

rows = [150.0, 113.0, 76.0, 39.0]
conds = [("Inline", ["report pasted into context"]),
         ("TOC", ["section index only"]),
         ("TOC + rounding hint", ["“do not truncate digits”"]),
         ("TOC + report hint", ["“take the value exactly", "as printed there”"])]
reads = [("result.out", "pH = ", "2.751", HL_REPORT),
         ("sel.out", "", "2.75065", HL_OWN),
         ("q1_step1.sel", "", "2.75065", HL_OWN),
         ("result.out", "pH = ", "2.751", HL_REPORT)]
answers = [(True, "(B) 2.751", ""), (False, "(A) 2.750", "65"), (False, "(A) 2.750", "65"), (True, "(B) 2.751", "")]

for y, (name, sub), (fname, pre, val, hl), (ok, ans, dropped) in zip(rows, conds, reads, answers):
    t(CX[0], y + 5, name, size=8)
    for k, line in enumerate(sub):
        t(CX[0], y - 5 - 8.5 * k, line, size=7, color=DGRAY, style="italic")
    t(CX[1], y + 5, fname, size=6.6, font=MONO)
    xv = CX[1]
    if pre:
        t(xv, y - 6, pre, size=6.6, font=MONO)
        xv += 22
    t(xv, y - 6, val, size=6.6, font=MONO, hl=hl)
    txt = t(CX[2], y, ans, size=8)
    fig.canvas.draw()
    xg = txt.get_window_extent().transformed(ax.transData.inverted()).x1 + 0.2
    if dropped:
        dtx = t(xg, y, dropped, size=8, color=MGRAY)
        fig.canvas.draw()
        xg = dtx.get_window_extent().transformed(ax.transData.inverted()).x1
    t(xg + 3, y, "✓" if ok else "✗", size=8.5, font=GLYPH, color=GREEN if ok else RED)
for yb in [(rows[i] + rows[i + 1]) / 2 for i in range(3)]:
    hline(BX0, DX1, yb + 3, 0.3, LGRAY)

# ------------------------------------------------------------------ (c) dot plots aligned with the rows
t(DX0 - 2, 178, "(c)", size=8.5, weight="bold")
panels = [(DX0 + 13, DX0 + 58, "skips report", 0), (DX0 + 76, DX0 + 114, "wrong once seen", 1)]
for x0, x1, title, idx in panels:
    t((x0 + x1) / 2, 178, title, size=7, ha="center")
    vmax = 60.0 if idx == 0 else 20.0
    ticks = [0, 30, 60] if idx == 0 else [0, 10, 20]
    for tv in ticks:
        xt = x0 + (x1 - x0) * tv / vmax
        ax.plot([xt, xt], [22, 166], color=LGRAY, lw=0.3, zorder=0)
        ax.plot([xt, xt], [20, 22.5], color=BLACK, lw=0.6)
        t(xt, 12, "%d%%" % tv, size=6.5, ha="center")
    ax.plot([x0, x1], [22, 22], color=BLACK, lw=0.6)
    for y, rate in zip(rows, RATES):
        for (vals, dy, filled) in [(rate[0], 3.0, True), (rate[1], -3.0, False)]:
            v = vals[idx]
            xd = x0 + (x1 - x0) * min(v, vmax) / vmax
            ax.plot([xd], [y + dy], marker="o", ms=3.4, mew=0.7, mec=BLACK, mfc=BLACK if filled else "white",
                    zorder=3, clip_on=False)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_evidence.pdf")
fig.savefig(out)
fig.savefig(os.path.splitext(out)[0] + ".png", dpi=260)
print("wrote", out)
