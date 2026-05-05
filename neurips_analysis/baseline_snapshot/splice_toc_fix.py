#!/usr/bin/env python3
"""
Splice the uncapped-TOC rerun (45 cap-affected items, 5 models) into the
baseline 200-item per-item correctness vectors and recompute every TOC-touching
aggregate that downstream tables / macros depend on.

Run AFTER all 5 rerun directories have completed:
    result/custom/neurips200_tocfix_uncapped_capaffected45_toc_opus46
    result/custom/neurips200_tocfix_uncapped_capaffected45_toc_sonnet46
    result/custom/neurips200_tocfix_uncapped_capaffected45_toc_gpt_5_4_
    result/custom/neurips200_tocfix_uncapped_capaffected45_toc_gpt51
    result/custom/neurips200_tocfix_uncapped_capaffected45_toc_gpt52

Outputs (written next to this script):
    spliced_per_item_correctness.csv      — 1000 rows, post-splice
    spliced_aggregates.csv                — 200-aggregate accuracy + tokens per model x method
    spliced_per_bin.csv                   — per-bin accuracy + tokens per model x method (TOC only)
    spliced_retention.csv                 — kept/gained/lost vs Direct baseline
    spliced_overlap_toc_raw100.csv        — Both/TOC-only/Raw-only/Neither
    macro_updates.tex                     — drop-in replacement chunk for results_macros.tex
"""

import csv, json
from pathlib import Path
from collections import defaultdict

REPO   = Path(__file__).resolve().parents[2]
SNAP   = Path(__file__).resolve().parent
RESULT = REPO / "result" / "custom"

NEW_DIRS = {
    "Claude Opus 4.6":   RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_opus46",
    "Claude Sonnet 4.6": RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_sonnet46",
    "GPT-5.4":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt_5_4_",
    "GPT-5.1":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt51",
    "GPT-5.2":           RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_gpt52",
}

UNION_45 = sorted(int(x) for x in (SNAP / "cap_affected_union45_ids.txt").read_text().split())

def load_baseline_correctness():
    """Load per-item baseline. Returns {model: {item_id: {method: 0/1 or None}}}."""
    items = defaultdict(dict)
    with (SNAP / "per_item_correctness_baseline.csv").open() as f:
        for r in csv.DictReader(f):
            m = r["model"]; idx = int(r["item_id"])
            items[m][idx] = {
                "direct":  None if r["direct_correct"]=="" else int(r["direct_correct"]),
                "cot":     None if r["cot_correct"]=="" else int(r["cot_correct"]),
                "toc":     None if r["toc_correct"]=="" else int(r["toc_correct"]),
                "raw100k": None if r["raw100k_correct"]=="" else int(r["raw100k_correct"]),
                "raw500k": None if r["raw500k_correct"]=="" else int(r["raw500k_correct"]),
            }
    return items

def load_new_toc(dir_path: Path):
    """Load post-rerun per-item TOC correctness from a results.jsonl."""
    out = {}
    rj = dir_path / "results.jsonl"
    if not rj.exists():
        raise FileNotFoundError(f"missing {rj} — has the run finished?")
    for line in rj.open():
        r = json.loads(line)
        # Field is "is_correct" in the results.jsonl produced by evaluate_custom.py.
        out[int(r["index"])] = int(bool(r.get("is_correct", False)))
    return out

def load_new_summary(dir_path: Path):
    """Total input/output tokens for the run."""
    sj = dir_path / "summary.json"
    if not sj.exists():
        return None
    s = json.load(sj.open())
    return {
        "input_tokens": s.get("total_input_tokens") or s.get("input_tokens") or 0,
        "output_tokens": s.get("total_output_tokens") or s.get("output_tokens") or 0,
        "n": s.get("num_questions") or s.get("total") or len(list(dir_path.glob("work_space/dataset_q_*"))),
    }

def load_per_item_size(model: str):
    """Per-item bin assignment for the model (None for Opus)."""
    if model not in {"Claude Sonnet 4.6", "GPT-5.1", "GPT-5.2", "GPT-5.4"}:
        return None
    out = {}
    with (SNAP / "per_item_output_size_per_model.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] == model:
                out[int(r["item_id"])] = r["bin"]
    return out

def main():
    baseline = load_baseline_correctness()
    spliced = {m: {i: dict(v) for i, v in baseline[m].items()} for m in baseline}

    # === Splice ===
    print(f"Splicing {len(UNION_45)} items per model...")
    new_per_model = {}
    new_summary_per_model = {}
    for m, d in NEW_DIRS.items():
        new = load_new_toc(d)
        new_per_model[m] = new
        new_summary_per_model[m] = load_new_summary(d)
        missing = [i for i in UNION_45 if i not in new]
        if missing:
            print(f"  WARN {m}: missing in new run: {missing}")
        for i in UNION_45:
            if i in new:
                spliced[m][i]["toc"] = new[i]

    # === Write spliced per-item ===
    out_csv = SNAP / "spliced_per_item_correctness.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","item_id","direct","cot","toc_old","toc_new",
                    "raw100k","raw500k"])
        for m in spliced:
            for i in sorted(spliced[m]):
                old = baseline[m][i].get("toc")
                new = spliced[m][i].get("toc")
                w.writerow([m, i,
                            "" if spliced[m][i]["direct"] is None else spliced[m][i]["direct"],
                            "" if spliced[m][i]["cot"]    is None else spliced[m][i]["cot"],
                            "" if old is None else old,
                            "" if new is None else new,
                            "" if spliced[m][i]["raw100k"] is None else spliced[m][i]["raw100k"],
                            "" if spliced[m][i]["raw500k"] is None else spliced[m][i]["raw500k"]])
    print(f"wrote {out_csv}")

    # === Recompute aggregates ===
    print("\n=== Spliced TOC headlines ===")
    headlines = {}
    for m in spliced:
        n_correct = sum(v["toc"] for v in spliced[m].values() if v["toc"] is not None)
        n_total = sum(1 for v in spliced[m].values() if v["toc"] is not None)
        headlines[m] = (n_correct, n_total)
        old_correct = sum(v["toc"] for v in baseline[m].values() if v["toc"] is not None)
        delta = n_correct - old_correct
        print(f"  {m:25s} TOC: {n_correct}/{n_total} = {100*n_correct/n_total:.1f}%  "
              f"(was {old_correct}/{n_total} = {100*old_correct/n_total:.1f}%, delta = {delta:+d} items)")

    # === Retention vs Direct ===
    print("\n=== Spliced retention (vs Direct, where Direct per-item is available) ===")
    retention = {}
    with (SNAP / "spliced_retention.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","kept","gained","lost","both_wrong","retention_pct","net_gain"])
        for m in spliced:
            d = spliced[m]
            if any(v["direct"] is None for v in d.values()):
                # Opus: Direct per-item missing — skip retention recomputation
                w.writerow([m,"-","-","-","-","-","-"])
                print(f"  {m:25s} Direct per-item not available — retention not recomputable")
                continue
            kept = sum(1 for v in d.values() if v["direct"]==1 and v["toc"]==1)
            lost = sum(1 for v in d.values() if v["direct"]==1 and v["toc"]==0)
            gained = sum(1 for v in d.values() if v["direct"]==0 and v["toc"]==1)
            both_wrong = sum(1 for v in d.values() if v["direct"]==0 and v["toc"]==0)
            ret = kept / (kept + lost) * 100 if (kept+lost) else 0
            net = (kept + gained) - (kept + lost)
            w.writerow([m, kept, gained, lost, both_wrong, f"{ret:.1f}%", net])
            print(f"  {m:25s} kept={kept} gained={gained} lost={lost} retention={ret:.1f}%  net={net:+d}")

    # === Overlap TOC vs Raw100k ===
    print("\n=== Spliced overlap (TOC vs Raw100k) ===")
    with (SNAP / "spliced_overlap_toc_raw100.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","TOC_right","Raw100k_right","Both","TOC_only","Raw_only","Neither"])
        for m in spliced:
            d = spliced[m]
            if any(v["raw100k"] is None for v in d.values()):
                w.writerow([m,"-","-","-","-","-","-"])
                print(f"  {m:25s} Raw100k per-item not available")
                continue
            both = sum(1 for v in d.values() if v["toc"]==1 and v["raw100k"]==1)
            toc_only = sum(1 for v in d.values() if v["toc"]==1 and v["raw100k"]==0)
            raw_only = sum(1 for v in d.values() if v["toc"]==0 and v["raw100k"]==1)
            neither = sum(1 for v in d.values() if v["toc"]==0 and v["raw100k"]==0)
            t = both + toc_only
            r = both + raw_only
            w.writerow([m, t, r, both, toc_only, raw_only, neither])
            print(f"  {m:25s} TOC={t} Raw100k={r}  Both={both} TOC-only={toc_only} Raw-only={raw_only} Neither={neither}")

    # === Per-bin TOC recompute (4 models with size data) ===
    print("\n=== Spliced per-bin TOC accuracy ===")
    with (SNAP / "spliced_per_bin.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","bin","n_items","toc_correct","toc_accuracy_pct"])
        for m in spliced:
            sizes = load_per_item_size(m)
            if sizes is None:
                continue
            bin_count = defaultdict(lambda: [0,0])  # bin -> [n, correct]
            for i in spliced[m]:
                if spliced[m][i]["toc"] is None: continue
                b = sizes.get(i, "?")
                bin_count[b][0] += 1
                bin_count[b][1] += spliced[m][i]["toc"]
            for b in ["<=10k","10-20k","20-50k","50-100k","100-250k","250-500k",">500k"]:
                n, c = bin_count.get(b, [0,0])
                if n == 0: continue
                w.writerow([m, b, n, c, f"{100*c/n:.1f}%"])

    # === Token aggregates ===
    print("\n=== Spliced token aggregates ===")
    # Old aggregates from baseline (for reference)
    old_agg = {}
    with (SNAP / "aggregate_tokens_baseline.csv").open() as f:
        for r in csv.DictReader(f):
            if r["method"] == "Agent-TOC":
                old_agg[r["model"]] = float(r["total_input_tokens_M"])
    # New aggregate = old aggregate – (estimate of cap-affected portion in old) + new run total
    # Simpler approach: just report new run total + old non-cap-affected portion (approximated from per-bin tables)
    # For now report side-by-side for downstream macro update
    print(f"  {'model':25s} {'old_TOC_input_M':>17s} {'new_45_input_M':>17s} {'~old_excl_45_M':>17s} {'spliced_total_M':>17s}")
    for m in spliced:
        old = old_agg.get(m, None)
        nm = new_summary_per_model[m]
        if nm is None or old is None:
            print(f"  {m:25s} (incomplete)")
            continue
        new_tok_M = nm["input_tokens"] / 1e6
        # Estimate of old cap-affected token usage: sum from per-bin tables for cap-affected bins
        # (For Opus we don't have per-bin, so leave as None.)
        old_capaffected_M = None
        with (SNAP / "per_bin_baseline.csv").open() as f:
            tot_chars = 0; have_data = False
            for r in csv.DictReader(f):
                if r["model"] == m and r["method"] == "Agent-TOC" and r["bin"] in ("250-500k", ">500k"):
                    tot_chars += int(r["n_items"]) * int(r["mean_input_tokens_per_item"])
                    have_data = True
            if have_data:
                old_capaffected_M = tot_chars / 1e6
        old_excl_M = old - old_capaffected_M if old_capaffected_M is not None else None
        spliced_tot = (old_excl_M + new_tok_M) if old_excl_M is not None else None
        print(f"  {m:25s} {old:>17.2f} {new_tok_M:>17.2f} "
              f"{(f'{old_excl_M:.2f}' if old_excl_M else '-'):>17s} "
              f"{(f'{spliced_tot:.2f}' if spliced_tot else '-'):>17s}")

    # === Macro update template ===
    print("\nWriting macro update template...")
    with (SNAP / "macro_updates.tex").open("w") as f:
        f.write("% Macro updates from TOC fix splice. Hand-merge into results_macros.tex.\n")
        f.write("% Generated by splice_toc_fix.py\n\n")
        for m, (correct, total) in headlines.items():
            macro = {
                "Claude Opus 4.6":   "OpusTOC",
                "Claude Sonnet 4.6": "SonnetTOC",
                "GPT-5.4":           "GPTFiveFourTOC",
                "GPT-5.2":           "GPTFiveTwoTOC",
                "GPT-5.1":           "GPTFiveOneTOC",
            }[m]
            f.write(f"\\renewcommand{{\\{macro}}}{{{100*correct/total:.1f}\\%}}  % was: see baseline\n")
    print(f"wrote {SNAP / 'macro_updates.tex'}")
    print("\nNext steps:")
    print("  1. Open spliced_per_item_correctness.csv and spot-check toc_old vs toc_new on the 45 items.")
    print("  2. Hand-merge macro_updates.tex into paper_neurips_ed/results_macros.tex.")
    print("  3. Update per-bin tables D.1-D.5 cap-affected rows (250-500k, >500k) from spliced_per_bin.csv.")
    print("  4. Update Table 3 tab:large-output-access from spliced_per_bin.csv (cap-affected aggregate).")
    print("  5. Recompile.")

if __name__ == "__main__":
    main()
