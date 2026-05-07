#!/usr/bin/env python3
"""
Splice the Opus 4.6 Raw100k rerun (per-item correctness + per-item input tokens)
into the per-item correctness table and compute Opus rows for
Table tab:capaffected-toc-verification.

Inputs:
    result/custom/neurips200_opusrawfix_raw100k_opus46/{results.jsonl,summary.json,work_space/}
    result/custom/neurips200_tocfix_uncapped_capaffected45_toc_opus46/   (for TOC/45 tokens)
    spliced_per_item_correctness.csv                                       (for Opus TOC per-item)
    cap_affected_union45_ids.txt                                           (UNION_45)

Outputs:
    spliced_per_item_correctness.csv  -- Opus raw100k_correct column populated
    opus_raw_fix_summary.txt          -- table values
"""
import csv, json
from pathlib import Path

REPO   = Path(__file__).resolve().parents[2]
SNAP   = Path(__file__).resolve().parent
RESULT = REPO / "result" / "custom"

OPUS_RAW_DIR = RESULT / "neurips200_opusrawfix_raw100k_opus46"
OPUS_TOC_DIR = RESULT / "neurips200_tocfix_uncapped_capaffected45_toc_opus46"

UNION_45 = sorted(int(x) for x in (SNAP / "cap_affected_union45_ids.txt").read_text().split())
ALL_IDS  = list(range(1, 201))
NOT_UNION = [i for i in ALL_IDS if i not in set(UNION_45)]

def get_log_path(run_dir: Path, idx: int):
    qdir = run_dir / "work_space" / f"dataset_q_{idx:05d}" / "logs"
    if not qdir.exists():
        return None
    files = sorted(qdir.glob("*.jsonl"))
    return files[-1] if files else None

def per_item_input_tokens(run_dir: Path, id_map=None):
    """Sum per-item assistant input_tokens from chat logs.

    id_map: optional dict subset_idx -> original_id. If provided, the run_dir
    workspace is assumed to be named by subset index, and we return a dict
    keyed by ORIGINAL id. If None, workspace is assumed to be named by
    original id (full 200-item run).
    """
    out = {}
    if id_map is not None:
        for sub_idx, orig_id in id_map.items():
            lp = get_log_path(run_dir, sub_idx)
            if lp is None:
                continue
            total = _sum_log_input_tokens(lp)
            out[orig_id] = total
        return out
    for idx in ALL_IDS:
        lp = get_log_path(run_dir, idx)
        if lp is None:
            continue
        out[idx] = _sum_log_input_tokens(lp)
    return out

def _sum_log_input_tokens(lp: Path):
    total = 0
    with lp.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            u = e.get("usage")
            if not isinstance(u, dict):
                continue
            if e.get("action") in ("assistant", "aux_llm_usage"):
                total += int(u.get("input_tokens") or 0)
    return total

def load_subset_id_map(jsonl_path: Path):
    """Map subset workspace index (1..N) -> original dataset id."""
    out = {}
    for i, line in enumerate(jsonl_path.open(), 1):
        r = json.loads(line)
        out[i] = int(r["id"])
    return out

def per_item_correct(run_dir: Path):
    out = {}
    for line in (run_dir / "results.jsonl").open():
        r = json.loads(line)
        out[int(r["index"])] = int(bool(r.get("is_correct")))
    return out

def fmt_pct(num, den):
    return f"{100*num/den:.1f}%" if den else "—"

def main():
    print("Loading Opus Raw100k per-item correctness...")
    raw_correct = per_item_correct(OPUS_RAW_DIR)
    print(f"  loaded {len(raw_correct)} items, {sum(raw_correct.values())} correct, "
          f"{100*sum(raw_correct.values())/len(raw_correct):.1f}%")

    print("Loading Opus Raw100k per-item input tokens (chat logs)...")
    raw_tokens = per_item_input_tokens(OPUS_RAW_DIR)
    raw_total_tok = sum(raw_tokens.values())
    print(f"  loaded {len(raw_tokens)} items, total input tokens = {raw_total_tok/1e6:.2f}M")

    print("Loading Opus TOC 45-rerun per-item input tokens (subset workspace -> original id)...")
    subset_dataset = REPO / "datasets" / "neurips_dataset_200_clean_capaffected_union45.jsonl"
    id_map = load_subset_id_map(subset_dataset)
    toc_tokens_45 = per_item_input_tokens(OPUS_TOC_DIR, id_map=id_map)
    print(f"  loaded {len(toc_tokens_45)} items, total = {sum(toc_tokens_45.values())/1e6:.2f}M")

    # Opus TOC per-item correctness (post-splice)
    print("Loading Opus TOC per-item correctness from spliced_per_item_correctness.csv...")
    toc_correct = {}
    with (SNAP / "spliced_per_item_correctness.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] != "Claude Opus 4.6":
                continue
            i = int(r["item_id"])
            if r["toc_new"] != "":
                toc_correct[i] = int(r["toc_new"])
    print(f"  loaded {len(toc_correct)} items, {sum(toc_correct.values())} correct")

    # Sonnet canonical small (162 items where Sonnet TOC run1 output <=100k);
    # used as the "canonical reference" partition cited in the table caption.
    sonnet_canonical_small = []
    with (SNAP / "canonical_output_sizes_sonnet_run1.csv").open() as f:
        for r in csv.DictReader(f):
            if int(r["cap_affected"]) == 0:
                sonnet_canonical_small.append(int(r["item_id"]))
    sonnet_canonical_small = sorted(sonnet_canonical_small)

    # Aggregate on slices used by tab:capaffected-toc-verification
    def agg(ids, label):
        rc = sum(raw_correct.get(i, 0) for i in ids if i in raw_correct)
        rn = sum(1 for i in ids if i in raw_correct)
        rt = sum(raw_tokens.get(i, 0) for i in ids)
        rt_n = sum(1 for i in ids if i in raw_tokens)
        tc = sum(toc_correct.get(i, 0) for i in ids if i in toc_correct)
        tn = sum(1 for i in ids if i in toc_correct)
        toc_45_t = sum(toc_tokens_45.get(i, 0) for i in ids if i in toc_tokens_45)
        toc_45_n = sum(1 for i in ids if i in toc_tokens_45)
        return {
            "label": label,
            "n": len(ids),
            "raw_correct": rc, "raw_n": rn, "raw_acc_pct": 100*rc/rn if rn else None,
            "raw_total_tok": rt, "raw_tok_per_item": rt/rt_n if rt_n else None,
            "toc_correct": tc, "toc_n": tn, "toc_acc_pct": 100*tc/tn if tn else None,
            "toc_45_total_tok": toc_45_t, "toc_45_n": toc_45_n,
        }

    canonical_small = agg(sonnet_canonical_small, "small-output bulk (Sonnet canonical, n=162)")
    not_union = agg(NOT_UNION, "small-output bulk (NOT in UNION_45, n=155)")
    union45  = agg(UNION_45, "long-output tail (UNION_45, n=45)")
    all200   = agg(ALL_IDS, "all 200")

    # TOC tokens for 45 set come directly from the OPUS TOC rerun
    toc_45_total = sum(toc_tokens_45.values())
    toc_45_per_item = toc_45_total / 45

    # Estimated Opus TOC tokens on small-bulk slice.
    # OG opus_toc dir is deleted, so we use the splice approach:
    #   Spliced \OpusTOCTokens = 23.76M (estimate) = 15.17M (cap-unaffected, NOT_UNION) + 8.59M (new 45-rerun)
    # For Sonnet canonical small (162 = 155 NOT_UNION + 7 overlap items), the 7 overlap
    # items have measured TOC tokens from the new opus TOC rerun.
    SPLICE_TOC_TOTAL_M = 23.76         # \OpusTOCTokens
    NEW_TOC_45_TOTAL_M = sum(toc_tokens_45.values()) / 1e6
    EST_TOC_NOT_UNION_M = SPLICE_TOC_TOTAL_M - NEW_TOC_45_TOTAL_M  # ~15.17M for 155 items
    overlap_ids = sorted(set(sonnet_canonical_small) & set(UNION_45))
    overlap_toc_tok = sum(toc_tokens_45.get(i, 0) for i in overlap_ids)
    EST_TOC_CANONICAL_M = EST_TOC_NOT_UNION_M + overlap_toc_tok / 1e6
    canonical_toc_per_item_est = EST_TOC_CANONICAL_M * 1e6 / 162

    out_lines = []
    out_lines.append("=== Opus 4.6 Raw100k splice — table cells for tab:capaffected-toc-verification ===\n")
    for slice_ in (canonical_small, not_union, union45, all200):
        out_lines.append(f"\n[{slice_['label']}]")
        out_lines.append(f"  Raw100k acc        : {slice_['raw_correct']}/{slice_['raw_n']} = {slice_['raw_acc_pct']:.2f}%")
        out_lines.append(f"  Raw100k tok/q (mean): {slice_['raw_tok_per_item']:.0f}  (total {slice_['raw_total_tok']/1e6:.3f}M)")
        if slice_['toc_acc_pct'] is not None:
            out_lines.append(f"  TOC     acc        : {slice_['toc_correct']}/{slice_['toc_n']} = {slice_['toc_acc_pct']:.2f}%")

    out_lines.append("\n[Long tail (45 items in UNION_45) — TABLE ROW VALUES]")
    out_lines.append(f"  TOC acc      : {union45['toc_acc_pct']:.1f}%  ({union45['toc_correct']}/45)")
    out_lines.append(f"  Raw100k acc  : {union45['raw_acc_pct']:.1f}%  ({union45['raw_correct']}/45)")
    out_lines.append(f"  delta acc    : {union45['toc_acc_pct']-union45['raw_acc_pct']:+.1f} pp  (TOC - Raw100k)")
    out_lines.append(f"  TOC tok/q    : {sum(toc_tokens_45.values())/45:.0f}  (=8.59M/45 = 191K)")
    out_lines.append(f"  Raw100k tok/q: {union45['raw_tok_per_item']:.0f}")
    saving45 = 1 - (sum(toc_tokens_45.values())/45) / union45['raw_tok_per_item']
    out_lines.append(f"  TOC saving   : {100*saving45:.1f}%")

    out_lines.append("\n[Small-output bulk (162 items per Sonnet canonical reference) — TABLE ROW VALUES]")
    out_lines.append(f"  TOC acc      : {canonical_small['toc_acc_pct']:.1f}%  ({canonical_small['toc_correct']}/162)")
    out_lines.append(f"  Raw100k acc  : {canonical_small['raw_acc_pct']:.1f}%  ({canonical_small['raw_correct']}/162)")
    out_lines.append(f"  delta acc    : {canonical_small['toc_acc_pct']-canonical_small['raw_acc_pct']:+.1f} pp")
    out_lines.append(f"  TOC tok/q    : ~{canonical_toc_per_item_est:.0f} (estimated; OG opus_toc dir deleted)")
    out_lines.append(f"  Raw100k tok/q: {canonical_small['raw_tok_per_item']:.0f}")
    saving_small = 1 - canonical_toc_per_item_est / canonical_small['raw_tok_per_item']
    out_lines.append(f"  TOC saving   : ~{100*saving_small:.1f}% (estimated)")

    text = "\n".join(out_lines)
    print()
    print(text)
    (SNAP / "opus_raw_fix_summary.txt").write_text(text)
    print(f"\nWrote {SNAP / 'opus_raw_fix_summary.txt'}")

    # Also splice raw100k_correct into spliced_per_item_correctness.csv
    print("\nUpdating spliced_per_item_correctness.csv with raw100k_correct for Opus...")
    rows = []
    with (SNAP / "spliced_per_item_correctness.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["model"] == "Claude Opus 4.6":
            i = int(r["item_id"])
            if i in raw_correct and (r.get("raw100k") or "") == "":
                r["raw100k"] = str(raw_correct[i])
    with (SNAP / "spliced_per_item_correctness.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","item_id","direct","cot","toc_old","toc_new","raw100k","raw500k"])
        w.writeheader()
        w.writerows(rows)
    print("  done.")

if __name__ == "__main__":
    main()
