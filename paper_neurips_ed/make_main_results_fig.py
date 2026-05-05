"""
Horizontal grouped bar chart of PHREEQC-MCQ-200 main results (Table 1)
with Wilson 95% confidence intervals on every bar.
"""
import matplotlib.pyplot as plt
import numpy as np


def wilson_ci(p_pct, n=200, z=1.96):
    """Wilson 95% CI for a binomial proportion. Returns (lo_pct, hi_pct)."""
    p = p_pct / 100.0
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (center - half) * 100, (center + half) * 100


# Data from Table 1 (tab:main-results), n=200
models = ['Claude Opus 4.6', 'Claude Sonnet 4.6', 'GPT-5.4',
          'GPT-5.2', 'GPT-5.1', 'Gemini 3 Flash Preview']
tiers  = ['Top', 'Top', 'Top', 'Mid', 'Mid', 'Below-mid']

data = {
    'Direct':         [42.0, 37.0, 27.0, 27.5, 39.0, 38.5],
    'CoT':            [38.0, 43.0, 35.0, 34.5, 35.0, 40.5],
    'Agent-TOC':      [83.5, 84.0, 72.5, 54.0, 54.0, 23.5],
    'Agent-Raw100k':  [85.0, 83.5, 70.0, 61.5, 63.5, 47.0],
}
colors = {
    'Direct':        '#bbbbbb',
    'CoT':           '#777777',
    'Agent-TOC':     '#1f77b4',
    'Agent-Raw100k': '#d62728',
}

fig, ax = plt.subplots(figsize=(9.0, 6.5))

bar_h = 0.18
gap_within_model = 0.02
gap_between_models = 0.55

n_models = len(models)
n_conditions = len(data)
group_height = n_conditions * bar_h + (n_conditions - 1) * gap_within_model
y_centers = np.arange(n_models) * (group_height + gap_between_models)

condition_offsets = (
    np.arange(n_conditions) * (bar_h + gap_within_model)
    - (group_height - bar_h) / 2
)

for i, (cond, color) in enumerate(colors.items()):
    values = np.array(data[cond])
    cis = [wilson_ci(v) for v in values]
    errs_lo = np.array([v - lo for v, (lo, _) in zip(values, cis)])
    errs_hi = np.array([hi - v for v, (_, hi) in zip(values, cis)])
    ax.barh(
        y_centers + condition_offsets[i],
        values,
        height=bar_h,
        xerr=[errs_lo, errs_hi],
        color=color,
        edgecolor='black',
        linewidth=0.4,
        label=cond,
        error_kw=dict(elinewidth=0.8, capsize=2.5, ecolor='black'),
    )

# Y-tick labels — model name with tier in parentheses
tier_color = {'Top': '#2ca02c', 'Mid': '#ff7f0e', 'Below-mid': '#d62728'}
ax.set_yticks(y_centers)
labels = [f'{m}\n({t} tier)' for m, t in zip(models, tiers)]
ax.set_yticklabels(labels, fontsize=9)
for tick_label, t in zip(ax.get_yticklabels(), tiers):
    tick_label.set_color('black')
ax.invert_yaxis()  # Top tier at top

# Tier-separator lines between rows
def midpoint(a, b):
    return (y_centers[a] + y_centers[b]) / 2

ax.axhline(midpoint(2, 3), color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
ax.axhline(midpoint(4, 5), color='gray', linestyle='--', linewidth=0.6, alpha=0.5)

# Tier shading (subtle bands)
def tier_span(a, b, color):
    top = y_centers[a] - group_height/2 - gap_between_models/4
    bot = y_centers[b] + group_height/2 + gap_between_models/4
    ax.axhspan(top, bot, alpha=0.05, color=color, zorder=0)

tier_span(0, 2, '#2ca02c')   # Top
tier_span(3, 4, '#ff7f0e')   # Mid
tier_span(5, 5, '#d62728')   # Below-mid

# Majority-label floor reference
ax.axvline(31.5, color='black', linestyle=':', linewidth=1, alpha=0.6)
ax.text(31.5, y_centers[-1] + group_height/2 + gap_between_models/2,
        ' majority floor (31.5%)',
        ha='left', va='top', fontsize=8, color='black', alpha=0.7)

ax.set_xlabel('Accuracy (%)', fontsize=10)
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 20))
ax.grid(axis='x', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
          frameon=False, fontsize=9, ncol=4)

plt.tight_layout()
plt.savefig('main_results.pdf', bbox_inches='tight')
plt.savefig('main_results.png', bbox_inches='tight', dpi=200)
print('Saved main_results.pdf and main_results.png')
