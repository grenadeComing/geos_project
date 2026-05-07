#!/usr/bin/env python3
"""
Build the calibrated (true-TOC) per-item correctness CSV.

Pipeline (CSV-based, transparent):
  1. Load per-item TOC-capped correctness for each model from
     `per_item_correctness_baseline.csv` (canonical baseline; already
     verified to match the failure-analysis back-deduction).
  2. Load the 45-item rerun results from each model's
     `neurips200_tocfix_uncapped_capaffected45_toc_<model>/results.jsonl`.
  3. Union splice: for every (model, item) where item is in the 45-item
     union, override the baseline with the rerun result. Leave the other
     155 items unchanged.
  4. Emit `calibrated_toc_per_item.csv` (long form, 1000 rows).

Run:  python3 calibrate_toc.py
"""

import csv, json
from pathlib import Path

SNAP   = Path(__file__).resolve().parent
REPO   = SNAP.parents[1]
RESULT = REPO / "result" / "custom"

RERUN_DIRS = {
    "Claude Opus 4.6":   RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_opus46",
    "Claude Sonnet 4.6": RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_sonnet46",
    "GPT-5.4":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt_5_4_",
    "GPT-5.1":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt51",
    "GPT-5.2":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt52",
}

MODEL_TO_RUN_NAME = {
    "Claude Opus 4.6":   "opus_toc",
    "Claude Sonnet 4.6": "sonnet_toc",
    "GPT-5.1":           "gpt51_toc",
    "GPT-5.2":           "gpt52_toc",
    "GPT-5.4":           "gpt54_toc",
}

UNION_45 = sorted(int(x) for x in (SNAP / "cap_affected_union45_ids.txt").read_text().split())


def load_baseline_toc_capped():
    """Step 1: per-(model, item) TOC-capped correctness from baseline CSV.

    Returns {model: {item_id: 0/1 or None}}. None means not recoverable.
    """
    out = {}
    with (SNAP / "per_item_correctness_baseline.csv").open() as f:
        for r in csv.DictReader(f):
            out.setdefault(r["model"], {})
            v = r["toc_correct"]
            out[r["model"]][int(r["item_id"])] = None if v == "" else int(v)
    return out


def load_rerun_results():
    """Step 2: per-(model, item) 45-item rerun correctness.

    Returns {model: {item_id: 0/1}}. Missing rerun dir -> empty dict.
    """
    out = {}
    for model, d in RERUN_DIRS.items():
        rj = d / "results.jsonl"
        out[model] = {}
        if not rj.exists():
            print(f"[warn] missing rerun results.jsonl for {model}: {rj}")
            continue
        for line in rj.open():
            r = json.loads(line)
            out[model][int(r["index"])] = int(bool(r.get("is_correct", False)))
    return out


def union_splice(baseline, rerun):
    """Step 3: override baseline with rerun for items in UNION_45."""
    calibrated = {}
    for model, items in baseline.items():
        calibrated[model] = {}
        for i, base_val in items.items():
            if i in set(UNION_45) and i in rerun.get(model, {}):
                calibrated[model][i] = rerun[model][i]
            else:
                calibrated[model][i] = base_val
    return calibrated


def write_long_csv(calibrated, out_path):
    """Step 4: long-format CSV (question_number, run_name, calibrated_correct)."""
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_number", "run_name", "calibrated_correct"])
        for model, items in calibrated.items():
            run = MODEL_TO_RUN_NAME[model]
            for i in sorted(items.keys()):
                v = items[i]
                w.writerow([i, run, "" if v is None else v])


def summarize(calibrated, baseline):
    print(f"\n{'Model':22s} | {'Baseline-capped':>16s} | {'Calibrated true-TOC':>20s} | {'Δ items':>9s}")
    for m in ["Claude Opus 4.6", "Claude Sonnet 4.6", "GPT-5.4", "GPT-5.2", "GPT-5.1"]:
        base = sum(v for v in baseline[m].values() if v is not None)
        cal  = sum(v for v in calibrated[m].values() if v is not None)
        n    = sum(1 for v in baseline[m].values() if v is not None)
        delta = cal - base
        print(f"{m:22s} | {base:>10d}/{n:<3d} | {cal:>14d}/{n:<3d} | {delta:+>9d}")


if __name__ == "__main__":
    baseline   = load_baseline_toc_capped()
    rerun      = load_rerun_results()
    calibrated = union_splice(baseline, rerun)

    out = SNAP / "calibrated_toc_per_item.csv"
    write_long_csv(calibrated, out)
    print(f"Wrote {out}")

    summarize(calibrated, baseline)
