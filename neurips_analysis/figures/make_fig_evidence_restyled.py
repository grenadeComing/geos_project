"""Figure 1 of the NAACL paper: same simulation, different evidence  (restyled copy of make_fig_evidence_v5.py).

Layout, content and data are unchanged from make_fig_evidence_v5.py; only the styling (colours, stroke weights,
fills vs borders, corner radii, arrow heads, bar shapes) is parameterised by a STYLE preset:

  A  "flat tinted cards"   borderless cards in Material-style container tints, clean azure + coral-red
  B  "tinted outlines"     the Excalidraw look of the earlier paper, but light hue-tinted strokes, teal + coral

Skip rates use all 158 verified items; closure rates use the 137 numeric items,
conditioned on the keyed value being seen. The figure caption states these denominators.

Usage: python make_fig_evidence_v5_restyled.py <out.pdf> [A|B]      (also writes <out>.png)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# fonts live next to the ORIGINAL script, not this copy
FIGDIR = "/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github/neurips_analysis/figures"
font_manager.fontManager.addfont(os.path.join(FIGDIR, "fonts", "Excalifont-Latin.ttf"))
for f in ["Times New Roman.ttf", "Times New Roman Bold.ttf", "Courier New.ttf", "Courier New Bold.ttf"]:
    p = os.path.join("/System/Library/Fonts/Supplemental", f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Times New Roman", "mathtext.fontset": "stix"})
HAND, MONO = "Excalifont", "Courier New"

# =============================================================================== style presets
# Roles: ink = primary text; sec = secondary hand-drawn text; head = column headers;
#        line = arrows / brace / fork; frame = outer frame;
#        neu_* = neutral cards (item, delivery conditions);
#        rep_* = simulator report -> correct;  own_* = agent's own file -> wrong
#          *_acc  = accent marks (check / cross glyphs)      *_txt = coloured values and notes (>= 4.5:1 on white)
#          *_cont = container fill                           *_edge = card outline (None = borderless)
#          *_sec  = secondary text on that container (same hue, not gray)
#        bar_* = failure-rate bars (both columns are failure rates -> warm family), track = full scale.
STYLES = {
    "A": dict(
        ink="#232933", sec="#5f6a77", hdr="#78818c", line="#b2b8bf", frame="#dadee3", frame_fc="#ffffff",
        neu_fc="#f4f6f9", neu_edge=None, neu_lw=0.0,
        rep_acc="#1570d1", rep_txt="#1467c2", rep_cont="#e2f0ff", rep_edge=None, rep_sec="#45668a",
        own_acc="#dd5139", own_txt="#c63a2a", own_cont="#ffebe5", own_edge=None, own_sec="#944a49",
        card_lw=0.0, badge_fc="cont", badge_edge=None, badge_lw=0.0, title_hue=True,
        bar={"5.1": "#fab8a8", "5.2": "#ef816a"}, track="#eef1f4", bar_r=0.3,
        r=1.0, badge_r=1.0, frame_r=1.8, frame_lw=0.6, line_lw=0.75, arrowhead=(2.6, 1.5), glyph_lw=1.45,
    ),
    "B": dict(
        ink="#212529", sec="#5c6670", hdr="#7b858f", line="#adb5bd", frame="#dee2e6", frame_fc="#ffffff",
        neu_fc="#ffffff", neu_edge="#ccd2d7", neu_lw=0.7,
        rep_acc="#029298", rep_txt="#017781", rep_cont="#e7fafb", rep_edge="#64c3c8", rep_sec="#3a666d",
        own_acc="#e15342", own_txt="#c83a35", own_cont="#fff3ef", own_edge="#f49383", own_sec="#894746",
        card_lw=0.8, badge_fc="white", badge_edge="edge", badge_lw=0.8, title_hue=False,
        bar={"5.1": "#fcc5ba", "5.2": "#f28e7d"}, track="#f1f3f5", bar_r=0.3,
        r=1.0, badge_r=1.0, frame_r=1.8, frame_lw=0.7, line_lw=0.8, arrowhead=(2.8, 1.6), glyph_lw=1.4,
    ),
}
VARIANT = (sys.argv[2] if len(sys.argv) > 2 else "A").upper()
S = STYLES[VARIANT]
INK, SEC, HEADC, LINE = S["ink"], S["sec"], S["hdr"], S["line"]
FAM = {
    "rep": dict(acc=S["rep_acc"], txt=S["rep_txt"], cont=S["rep_cont"], edge=S["rep_edge"], sec=S["rep_sec"]),
    "own": dict(acc=S["own_acc"], txt=S["own_txt"], cont=S["own_cont"], edge=S["own_edge"], sec=S["own_sec"]),
}

# --------------------------------------------------------------------- data (paper macros)
# per row: (skip report %, closure %) for GPT-5.1 and GPT-5.2
RATES = [
    ((0.0, 94.37), (0.0, 95.35)),     # inline (logged run)      closure = exact c|s from labels.jsonl
    ((53.0, 85.34), (60.3, 82.40)),   # TOC (three reruns)
    ((45.9, 81.82), (55.1, 81.48)),   # TOC + rounding hint
    ((1.6, 99.22), (0.0, 98.79)),     # TOC + report hint
]

W, H = 6.3, 2.75                       # inches; drawn in units of 0.1 in
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W * 10)
ax.set_ylim(0, H * 10)
ax.axis("off")


BOXES, TEXTS = [], []                  # recorded for the overflow check (CHECK=1)


def box(x0, y0, x1, y1, ec, fc, lw=1.0, r=None, z=1):
    r = S["r"] if r is None else r
    BOXES.append((x0, y0, x1, y1))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=%g" % r,
                                linewidth=lw if ec else 0, edgecolor=ec or "none", facecolor=fc, zorder=z))


def arrow(p0, p1, color=None, lw=None, z=2):
    hl, hw = S["arrowhead"]
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=%g,head_width=%g" % (hl, hw),
                                 color=color or LINE, lw=lw or S["line_lw"], shrinkA=0, shrinkB=0, zorder=z,
                                 capstyle="round", joinstyle="round"))


def text(x, y, s, size=7.4, font=None, color=None, weight="normal", ha="center", va="center", **kw):
    TEXTS.append(ax.text(x, y, s, fontsize=size, family=font or "Times New Roman", color=color or INK,
                         weight=weight, ha=ha, va=va, **kw))


def check(x, y, s=0.7, color=None):
    ax.plot([x - 0.9 * s, x - 0.25 * s, x + 1.0 * s], [y + 0.05 * s, y - 0.7 * s, y + 0.95 * s],
            color=color, lw=S["glyph_lw"], solid_capstyle="round", solid_joinstyle="round", zorder=4)


def cross(x, y, s=0.55, color=None):
    for dx in (-1, 1):
        ax.plot([x - s, x + s], [y - dx * s, y + dx * s], color=color, lw=S["glyph_lw"], solid_capstyle="round",
                zorder=4)


def brace(x, y0, y1, w, color=None, lw=None):
    ym = (y0 + y1) / 2
    t = np.linspace(0, 1, 60)
    for a, b in [(y0, ym), (y1, ym)]:
        yy = a + (b - a) * t
        xx = x + 0.45 * w * np.sin(np.pi * np.minimum(t, 0.5)) + 0.55 * w * np.clip((t - 0.8) / 0.2, 0, 1) ** 2
        ax.plot(xx, yy, color=color or LINE, lw=lw or S["line_lw"], solid_capstyle="round", zorder=2)


def card(x0, y0, x1, y1, fam, r=None):
    f = FAM[fam]
    box(x0, y0, x1, y1, f["edge"], f["cont"], lw=S["card_lw"], r=r)


def bar(x0, y0, w, h, fc, r):
    r = min(r, w / 2, h / 2)
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, linewidth=0,
                                edgecolor="none", facecolor=fc, zorder=2))


T, TB, HD, MO = 7.4, 7.7, 6.3, 7.0     # Times, Times bold, hand-drawn, mono (pt)
from matplotlib import patheffects
HBOLD = [patheffects.withStroke(linewidth=0.45, foreground="#232933")]   # faux bold for the hand-drawn column headers

# ------------------------------------------------------------------ frame and column headers
box(0.4, 0.4, W * 10 - 0.4, H * 10 - 0.4, S["frame"], S["frame_fc"], lw=S["frame_lw"], r=S["frame_r"], z=0)
L0, L1 = 1.3, 17.6                     # left panel
C0, C1 = 20.2, 32.2                    # condition boxes
R0, R1 = 33.7, 44.1                  # what the agent answers from
A0, A1 = 45.4, 49.1                    # answer badge
XM = 53.5                              # model tag
S0, SS = 51.6, 0.054                   # skips-report bars: start, units per % (nudged left: % labels cleared the frame by only 0.015 in)
E0, ES = 57.45, 0.158                 # wrong-once-seen bars: start, units per %
rows = [22.1, 16.2, 10.3, 4.4]
hh = 2.1
HY = 25.9
text((C0 + C1) / 2, HY, "delivery", size=HD, font=HAND, color=INK, path_effects=HBOLD)
text((R0 + R1) / 2, HY, "answers from", size=HD, font=HAND, color=INK, path_effects=HBOLD)
text((A0 + A1) / 2, HY, "answer", size=HD, font=HAND, color=INK, path_effects=HBOLD)
NS, NE = 55.3, 59.75                    # centres of the two number columns
text(NS, HY + 0.42, "skips", size=HD - 0.4, font=HAND, color=INK, path_effects=HBOLD)
text(NS, HY - 0.6, "report (%)", size=HD - 0.4, font=HAND, color=INK, path_effects=HBOLD)
text(NE, HY + 0.42, "wrong once", size=HD - 0.4, font=HAND, color=INK, path_effects=HBOLD)
text(NE, HY - 0.6, "seen (%)", size=HD - 0.4, font=HAND, color=INK, path_effects=HBOLD)

# ------------------------------------------------------------------ left: item and the two files
box(L0, 18.0, L1, 24.8, S["neu_edge"], S["neu_fc"], lw=S["neu_lw"])
text((L0 + L1) / 2, HY, "item", size=HD, font=HAND, color=INK, path_effects=HBOLD)
text(L0 + 0.8, 23.5, "Pure water + 0.001 mol H$_2$SO$_4$", size=T, ha="left")
text(L0 + 0.8, 21.6, "pH after reaction step 1?", size=T, ha="left")
for i, (lab, val) in enumerate([("A", "2.750"), ("B", "2.751"), ("C", "2.748"), ("D", "2.745")]):
    key = lab == "B"
    text(L0 + 0.8 + i * 3.85, 19.3, "%s %s" % (lab, val), size=T, ha="left", weight="bold" if key else "normal",
         color=FAM["rep"]["txt"] if key else INK)
text((L0 + L1) / 2, 16.4, "its simulation writes both files", size=HD, font=HAND, color=SEC)
F = (L1 - L0 - 0.8) / 2
fy0, fy1 = 3.4, 14.8
for fam, xc, title, l1, l2, l3, val in [
        ("rep", L0 + F / 2, "result.out", "simulator report", "rounded to the", "options' precision", "pH = 2.751"),
        ("own", L1 - F / 2, "sel.out", "agent's own file", "more digits than", "the options", "2.75065")]:
    f = FAM[fam]
    card(xc - F / 2, fy0, xc + F / 2, fy1, fam)
    text(xc, 13.1, title, size=MO, font=MONO, weight="bold", color=INK)
    text(xc, 11.0, l1, size=HD, font=HAND, color=f["sec"])
    text(xc, 9.0, l2, size=HD - 0.4, font=HAND, color=f["sec"])
    text(xc, 8.0, l3, size=HD - 0.4, font=HAND, color=f["sec"])
    text(xc, 5.4, val, size=MO + 0.4, font=MONO, color=f["txt"], weight="bold")

# brace and fork
ym = (fy0 + fy1) / 2
brace(L1 + 0.25, fy0, fy1, 0.7)
fork_x = 19.0
ax.plot([L1 + 0.95, fork_x], [ym, ym], color=LINE, lw=S["line_lw"], solid_capstyle="round", zorder=2)
ax.plot([fork_x, fork_x], [rows[-1], rows[0]], color=LINE, lw=S["line_lw"], solid_capstyle="round", zorder=2)

# ------------------------------------------------------------------ rows
conds = [("Inline", "report in context"),
         ("TOC (retrieval)", "section index only"),
         ("TOC + rounding hint", "“round, don't truncate”"),
         ("TOC + report hint", "“answer from result.out”")]
reads = [("result.out", "pH = 2.751", "rep"),
         ("sel.out", "2.75065 → 2.750", "own"),
         ("q1_step1.sel", "2.75065 → 2.750", "own"),
         ("result.out", "pH = 2.751", "rep")]
answers = [("B", True), ("A", False), ("A", False), ("B", True)]
notes = [None, "never opens result.out", "told to round, still truncates", None]

for y, (name, sub), (rlab, rval, fam), (ans, ok), note, rate in zip(rows, conds, reads, answers, notes, RATES):
    f = FAM[fam]
    arrow((fork_x, y), (C0 - 0.1, y))
    box(C0, y - hh, C1, y + hh, S["neu_edge"], S["neu_fc"], lw=S["neu_lw"])
    text((C0 + C1) / 2, y + 0.85, name, size=TB, weight="bold")
    text((C0 + C1) / 2, y - 1.1, sub, size=HD, font=HAND, color=SEC)
    arrow((C1, y), (R0 - 0.1, y))
    card(R0, y - hh, R1, y + hh, fam)
    text((R0 + R1) / 2, y + 0.9, rlab, size=MO - 0.4, font=MONO, color=f["sec"] if S["title_hue"] else INK)
    text((R0 + R1) / 2, y - 1.05, rval, size=MO, font=MONO, color=f["txt"], weight="bold")
    arrow((R1, y), (A0 - 0.1, y))
    bf = FAM["rep" if ok else "own"]
    fc = bf["cont"] if S["badge_fc"] == "cont" else S["badge_fc"]
    ec = bf["edge"] if S["badge_edge"] == "edge" else S["badge_edge"]
    box(A0, y - 1.7, A1, y + 1.7, ec, fc, lw=S["badge_lw"], r=S["badge_r"])
    text(A0 + 1.15, y, ans, size=8.8, weight="bold", color=INK)
    (check if ok else cross)(A0 + 2.5, y - 0.05, color="#2e9e57" if ok else "#d93a2b")   # green check, red cross
    if note:
        text((R0 + R1) / 2, y - hh - 0.85, note, size=HD - 0.2, font=HAND, color=FAM["own"]["txt"])
    # right panel: two models per row, numbers only (percent in the column header)
    ncol = INK
    for (skip, clo), dy, tag in zip(rate, (0.85, -0.85), ("5.1", "5.2")):
        yy = y + dy
        text(XM, yy, "GPT-" + tag, size=T - 1.0, ha="right", color=SEC)
        text(NS, yy, "%d" % round(skip), size=T, weight="bold", color=ncol)
        text(NE, yy, "%d" % round(100 - clo), size=T, weight="bold", color=ncol)

text((NS + NE) / 2, 1.5, "see caption", size=HD - 0.7, font=HAND, color=HEADC)

if os.environ.get("CHECK"):
    # every text must sit inside the smallest box containing its anchor with >= 0.35 units (0.035 in) clearance,
    # stay >= 0.5 units inside the frame, and not overlap another text
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bbs = []
    for t in TEXTS:
        e = t.get_window_extent()
        (a, b), (c, d) = inv.transform([(e.x0, e.y0), (e.x1, e.y1)])
        bbs.append((a, b, c, d, t.get_text()))
    frame = BOXES[0]
    for a, b, c, d, s_ in bbs:
        cx, cy = (a + c) / 2, (b + d) / 2
        inside = [bx for bx in BOXES[1:] if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]]
        if inside:
            bx = min(inside, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
            gap = min(a - bx[0], bx[2] - c)
            if gap < 0.35:
                print("TIGHT in box  %-28r horizontal clearance %.2f" % (s_, gap))
        gap = min(a - frame[0], frame[2] - c, b - frame[1], frame[3] - d)
        if gap < 0.5:
            print("TIGHT vs frame %-28r clearance %.2f" % (s_, gap))
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            p_, q_ = bbs[i], bbs[j]
            dx = max(p_[0], q_[0]) - min(p_[2], q_[2])
            dy = max(p_[1], q_[1]) - min(p_[3], q_[3])
            if dx < 0.25 and dy < 0.0:
                print("CLOSE texts %r / %r  dx=%.2f" % (p_[4], q_[4], dx))

out = sys.argv[1] if len(sys.argv) > 1 else "variant_%s.pdf" % VARIANT
fig.savefig(out)
fig.savefig(os.path.splitext(out)[0] + ".png", dpi=220)
print("wrote", out)
