#!/usr/bin/env python3
"""
Apply the 45-item uncapped TOC splice to each TOC variance rerun, so the
variance table tab:toc-rerun-variance reports SD/Range under the same
configuration as the spliced headline TOC accuracy.

Inputs (per model):
    variance_runs[i].results.jsonl + .summary.json    (3 reruns at capped TOC)
    uncapped_45[i].results.jsonl + .summary.json      (single 45-item uncapped)

Spliced rerun N accuracy = (variance_N correct on items NOT in UNION_45)
                          + (single uncapped rerun's UNION_45 correct)
Spliced rerun N tokens   = (variance_N total - variance_N's UNION_45 tokens)
                          + uncapped_45's total input tokens

Outputs:
    spliced_variance_summary.txt  — table of values to copy into macros
"""
import csv, json, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAP = Path(__file__).resolve().parent
RESULT = REPO / "result" / "custom"

UNION_45 = sorted(int(x) for x in (SNAP / "cap_affected_union45_ids.txt").read_text().split())
UNION_SET = set(UNION_45)

# Variance reruns per model (3 each).
# After running the Opus variance reruns (neurips200_opus_tocvariance_run{1,2,3}_toc_opus46),
# uncomment the Opus entry below and re-run this script to get spliced Opus stats.
VARIANCE = {
    "Claude Opus 4.6": [
        RESULT / "neurips200_opus_tocvariance_run1_toc_opus46",
        RESULT / "neurips200_opus_tocvariance_run2_toc_opus46",
        RESULT / "neurips200_opus_tocvariance_run3_toc_opus46",
    ],
    "Claude Sonnet 4.6": [
        RESULT / "neurips200_clean_rerun_20260504_sonnet_run1_toc_sonnet46",
        RESULT / "neurips200_clean_rerun_20260504_sonnet_run2_toc_sonnet46",
        RESULT / "neurips200_clean_rerun_20260504_sonnet_run3_toc_sonnet46",
    ],
    "GPT-5.4": [
        RESULT / "neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt_5_4_",
        RESULT / "neurips200_clean_rerun_20260503_alltoc_run2_redo_gpt54_gpt52_toc_gpt_5_4_",
        RESULT / "neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt_5_4_",
    ],
    "GPT-5.2": [
        RESULT / "neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt52",
        RESULT / "neurips200_clean_rerun_20260503_alltoc_run2_redo_gpt54_gpt52_toc_gpt52",
        RESULT / "neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt52",
    ],
    "GPT-5.1": [
        RESULT / "neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt51",
        RESULT / "neurips200_clean_rerun_20260503_alltoc_run2_toc_gpt51",
        RESULT / "neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt51",
    ],
}

# Single 45-item uncapped TOC reruns (the splice source)
UNCAPPED_45 = {
    "Claude Opus 4.6":   RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_opus46",
    "Claude Sonnet 4.6": RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_sonnet46",
    "GPT-5.4":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt_5_4_",
    "GPT-5.2":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt52",
    "GPT-5.1":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt51",
}

# Spliced original (post-fix headline) per model
ORIG_ACC = {
    "Claude Opus 4.6":   83.5,
    "Claude Sonnet 4.6": 84.0,
    "GPT-5.4":           72.5,
    "GPT-5.2":           54.0,
    "GPT-5.1":           54.0,
}
ORIG_TOK_M = {
    "Claude Opus 4.6":   23.76,
    "Claude Sonnet 4.6": 22.14,
    "GPT-5.4":           12.69,
    "GPT-5.2":           13.92,
    "GPT-5.1":            9.51,
}

def per_item_correct(rj):
    return {int(json.loads(l)["index"]): int(bool(json.loads(l).get("is_correct"))) for l in rj.open()}

def per_item_input_tokens(run_dir, by_subset_idx_to_orig=None):
    """Sum assistant + aux_llm_usage input tokens per item."""
    out = {}
    if by_subset_idx_to_orig:
        # Workspace named by subset index; map to original id
        for sub_idx, orig in by_subset_idx_to_orig.items():
            log = next((Path(run_dir) / "work_space" / f"dataset_q_{sub_idx:05d}" / "logs").glob("*.jsonl"), None)
            if log is None: continue
            out[orig] = _sum_log(log)
    else:
        for d in sorted((run_dir / "work_space").glob("dataset_q_*")):
            idx = int(d.name.split("_")[-1])
            log = next((d / "logs").glob("*.jsonl"), None)
            if log is None: continue
            out[idx] = _sum_log(log)
    return out

def _sum_log(log):
    total = 0
    with log.open() as f:
        for line in f:
            try: e = json.loads(line)
            except: continue
            u = e.get("usage")
            if isinstance(u, dict) and e.get("action") in ("assistant", "aux_llm_usage"):
                total += int(u.get("input_tokens") or 0)
    return total

def load_subset_id_map(jsonl_path):
    out = {}
    for i, line in enumerate(jsonl_path.open(), 1):
        out[i] = int(json.loads(line)["id"])
    return out

# ---- Compute ----
print(f"{'Model':<22} {'OrigAcc':>8} {'Run1':>8} {'Run2':>8} {'Run3':>8} {'AccDiff':<22} {'SD':>5} {'Rng':>5}")
results = {}
for m, runs in VARIANCE.items():
    # Single 45-rerun (uncapped) per-item
    uc_dir = UNCAPPED_45[m]
    subset_map = load_subset_id_map(REPO / "datasets" / "neurips_dataset_200_clean_capaffected_union45.jsonl")
    uc_correct = per_item_correct(uc_dir / "results.jsonl")  # keyed by original id
    uc_summary = json.load((uc_dir / "summary.json").open())
    uc_total_tok_M = uc_summary["total_input_tokens"] / 1e6

    spliced_accs = []
    spliced_toks_M = []
    for run_dir in runs:
        var_correct = per_item_correct(run_dir / "results.jsonl")
        var_summary = json.load((run_dir / "summary.json").open())
        var_total_tok_M = var_summary["total_input_tokens"] / 1e6
        # NOT_UNION correct from variance run; UNION correct from uncapped 45-rerun
        not_union_correct = sum(var_correct.get(i, 0) for i in range(1, 201) if i not in UNION_SET)
        union_correct_const = sum(uc_correct.get(i, 0) for i in UNION_45)
        spliced_correct = not_union_correct + union_correct_const
        spliced_accs.append(100 * spliced_correct / 200)

        # Tokens: sum variance run's per-item NOT_UNION + uncapped run total
        var_tokens = per_item_input_tokens(run_dir)
        not_union_tok_M = sum(var_tokens.get(i, 0) for i in range(1, 201) if i not in UNION_SET) / 1e6
        spliced_tok_M = not_union_tok_M + uc_total_tok_M
        spliced_toks_M.append(spliced_tok_M)

    accs = spliced_accs
    toks = spliced_toks_M
    o_acc = ORIG_ACC[m]
    o_tok = ORIG_TOK_M[m]
    diffs = [a - o_acc for a in accs]
    tdiffs = [t - o_tok for t in toks]
    acc_sd = statistics.stdev(accs) if len(accs) > 1 else 0
    tok_sd = statistics.stdev(toks) if len(toks) > 1 else 0
    acc_rng = max(accs) - min(accs)
    tok_rng = max(toks) - min(toks)
    results[m] = dict(accs=accs, toks=toks, diffs=diffs, tdiffs=tdiffs,
                      acc_sd=acc_sd, tok_sd=tok_sd, acc_rng=acc_rng, tok_rng=tok_rng,
                      orig_acc=o_acc, orig_tok=o_tok)
    diff_str = ', '.join(f"{d:+.1f}" for d in diffs)
    print(f"{m:<22} {o_acc:>7.1f}% {accs[0]:>7.1f}% {accs[1]:>7.1f}% {accs[2]:>7.1f}%  {diff_str:<20}  {acc_sd:>4.2f}  {acc_rng:>4.2f}")

print()
print(f"{'Model':<22} {'OrigTokM':>9} {'Run1':>8} {'Run2':>8} {'Run3':>8} {'TokDiff':<22} {'TokSD':>5} {'TokRng':>6}")
for m, r in results.items():
    diff_str = ', '.join(f"{d:+.2f}" for d in r['tdiffs'])
    print(f"{m:<22} {r['orig_tok']:>8.2f}M {r['toks'][0]:>7.2f}M {r['toks'][1]:>7.2f}M {r['toks'][2]:>7.2f}M  {diff_str:<20}  {r['tok_sd']:>4.2f}  {r['tok_rng']:>5.2f}")

# Write macro updates
out_path = SNAP / "spliced_variance_macros.tex"
SHORT = {"Claude Opus 4.6":"Opus","Claude Sonnet 4.6":"Sonnet","GPT-5.4":"GPTFiveFour","GPT-5.2":"GPTFiveTwo","GPT-5.1":"GPTFiveOne"}
with out_path.open("w") as f:
    f.write("% Macro updates after applying 45-item uncapped TOC splice to variance reruns.\n")
    f.write("% Generated by splice_variance_reruns.py — hand-merge into results_macros.tex.\n\n")
    for m, r in results.items():
        s = SHORT[m]
        f.write(f"% {m}\n")
        f.write(f"\\renewcommand{{\\{s}TOCRerunValues}}{{{r['accs'][0]:.1f}\\%, {r['accs'][1]:.1f}\\%, {r['accs'][2]:.1f}\\%}}\n")
        f.write(f"\\renewcommand{{\\{s}TOCRerunAccDiff}}{{{r['diffs'][0]:+.1f}, {r['diffs'][1]:+.1f}, {r['diffs'][2]:+.1f}}}\n")
        f.write(f"\\renewcommand{{\\{s}TOCRerunSD}}{{{r['acc_sd']:.1f}}}\n")
        f.write(f"\\renewcommand{{\\{s}TOCRerunRange}}{{{r['acc_rng']:.1f}}}\n")
        f.write(f"\\renewcommand{{\\{s}TOCRerunTokens}}{{{r['toks'][0]:.2f}M, {r['toks'][1]:.2f}M, {r['toks'][2]:.2f}M}}\n")
        f.write(f"\\renewcommand{{\\{s}TOCRerunTokDiff}}{{{r['tdiffs'][0]:+.2f}, {r['tdiffs'][1]:+.2f}, {r['tdiffs'][2]:+.2f}}}\n")
        f.write(f"\\renewcommand{{\\{s}TOCRerunTokSD}}{{{r['tok_sd']:.2f}}}\n")
        f.write(f"\\renewcommand{{\\{s}TOCRerunTokRange}}{{{r['tok_rng']:.2f}}}\n\n")
print(f"\nWrote {out_path}")
