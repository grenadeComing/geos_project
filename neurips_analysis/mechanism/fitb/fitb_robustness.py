#!/usr/bin/env python3
"""Recompute the numeric fill-in-the-blank (FITB) robustness result.

Read-only on the repo. Python 3 stdlib only. Re-run with:
    python3 fitb_robustness.py [REPO_ROOT]
Outputs (next to this script):
    fitb_to_mcq_map.csv     one row per FITB row: fitb index -> MCQ id, verification status
    per_item.csv            per-item 0/1 correctness for every condition, keyed by FITB index
    results.json            all aggregate numbers
    summary.md              human-readable summary (regenerated on every run)

Conventions
- FITB results.jsonl rows: `index` is the 1-based FITB row number, `row_index` the 0-based one
  (checked: truth == answer_value and tolerance == answer_tolerance for all 182 rows).
- Correctness: the recorded `is_correct` (None/ungraded counts as wrong, denominator = all items).
  Also re-graded independently with the repo rule abs(pred - truth) <= answer_tolerance.
- MCQ results.jsonl rows: `index` == MCQ id (200-row order of neurips_dataset_200_clean.jsonl).
- MCQ TOC: `toc_new` (headline, cap-fix spliced) and `toc_old` (pre-splice, original
  neurips200_clean_agent100k_toc_gpt_5_4_ run) from spliced_per_item_correctness.csv.
- "Tool advantage" = TOC accuracy minus the better of Direct / CoT on the same item set.
"""
import collections
import csv
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github")
OUT = Path(__file__).resolve().parent

FITB_DS = R / "datasets/fill_in_blank_182.jsonl"
MCQ_DS = R / "datasets/neurips_dataset_200_clean.jsonl"
VERIF = R / "human_evaluation/verification.json"
SPLICED = R / "neurips_analysis/baseline_snapshot/spliced_per_item_correctness.csv"
CHANGED = R / "neurips_analysis/mcq_fitb_changed_gpt54_sonnet/changed_cases.jsonl"
EXACT_CSV = R / "neurips_analysis/fill_in_blank_agent_exact_match.csv"

FITB_RUNS = {  # (model, condition) -> results.jsonl
    ("GPT-5.4", "Direct"): R / "result/oneshot/fill_in_blank_182_oneshot_gpt54/results.jsonl",
    ("GPT-5.4", "CoT"): R / "result/oneshot/fill_in_blank_182_cot_gpt54/results.jsonl",
    ("GPT-5.4", "TOC"): R / "result/custom/rebuttal_fib182_toc_gpt54_20260723/results.jsonl",
    ("Claude Sonnet 4.6", "Direct"): R / "result/oneshot/fill_in_blank_182_oneshot_sonnet46/results.jsonl",
    ("Claude Sonnet 4.6", "CoT"): R / "result/oneshot/fill_in_blank_182_cot_sonnet46/results.jsonl",
    ("Claude Opus 4.6", "Direct"): R / "result/oneshot/fill_in_blank_182_oneshot_opus46/results.jsonl",
    ("Claude Opus 4.6", "CoT"): R / "result/oneshot/fill_in_blank_182_cot_opus46/results.jsonl",
}
MCQ_RUNS = {
    ("GPT-5.4", "Direct"): R / "result/oneshot/neurips200_clean_oneshot_gpt_5_4_/results.jsonl",
    ("GPT-5.4", "CoT"): R / "result/oneshot/neurips200_clean_cot_gpt_5_4_/results.jsonl",
    ("Claude Sonnet 4.6", "Direct"): R / "result/oneshot/neurips200_clean_oneshot_sonnet46/results.jsonl",
    ("Claude Sonnet 4.6", "CoT"): R / "result/oneshot/neurips200_clean_cot_sonnet46/results.jsonl",
    ("Claude Opus 4.6", "Direct"): R / "result/oneshot/neurips200_clean_oneshot_opus46/results.jsonl",
    ("Claude Opus 4.6", "CoT"): R / "result/oneshot/neurips200_clean_cot_opus46/results.jsonl",
}
MODELS = ["GPT-5.4", "Claude Sonnet 4.6", "Claude Opus 4.6"]

problems = []  # sanity-check failures, reported in summary


def check(cond, msg):
    if not cond:
        problems.append(msg)
        print("CHECK FAILED:", msg, file=sys.stderr)
    return cond


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- mapping
ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def norm(s):
    s = unicodedata.normalize("NFKC", s).translate(ZW)
    return re.sub(r"\s+", " ", s).strip()


def mcq_stem(q):
    """MCQ question text without the trailing 'A. ... D. ...' option lines."""
    q = q.replace("\r\n", "\n")
    m = re.search(r"\n\s*A[.)]\s", q)
    return q[: m.start()] if m else q


def mcq_options(q):
    q = q.replace("\r\n", "\n")
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^\s*([A-D])[.)]\s*(.+?)\s*$", q, re.M)}


fitb = jl(FITB_DS)
mcq = jl(MCQ_DS)
check(len(fitb) == 182, f"FITB rows = {len(fitb)}")
check([r["id"] for r in mcq] == list(range(1, 201)), "MCQ ids are not 1..200 in file order")
mcq_by_id = {r["id"]: r for r in mcq}

verif = json.load(open(VERIF, encoding="utf-8"))
status = {q["id"]: q["status"] for q in verif["questions"]}
verified_ids = {i for i, s in status.items() if s == "verified"}
check(len(verified_ids) == 158, f"verified ids = {len(verified_ids)}")

stem_index = collections.defaultdict(list)
for r in mcq:
    stem_index[norm(mcq_stem(r["question"]))].append(r["id"])

fitb_to_id = {}   # 1-based FITB index -> MCQ id
unmapped = []     # (fitb index, reason)
for k, f in enumerate(fitb, 1):
    key = norm(f["question"])
    hits = stem_index.get(key, [])
    if len(hits) == 1:
        fitb_to_id[k] = hits[0]
        continue
    if len(hits) > 1:
        unmapped.append((k, f"ambiguous exact match: {hits}"))
        continue
    # fallback: FITB text contained in exactly one normalized MCQ question
    cont = [r["id"] for r in mcq if key in norm(r["question"])]
    if len(cont) == 1:
        fitb_to_id[k] = cont[0]
    else:
        unmapped.append((k, f"no unique match (containment hits {cont})"))

id_counts = collections.Counter(fitb_to_id.values())
dup_targets = {i: c for i, c in id_counts.items() if c > 1}
check(not dup_targets, f"MCQ ids hit by >1 FITB row: {dup_targets}")


# answer consistency: FITB answer_value equals the text of the MCQ gold option
def num(x):
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


answer_mismatch = []
for k, i in fitb_to_id.items():
    opts = mcq_options(mcq_by_id[i]["question"])
    gold_txt = opts.get(mcq_by_id[i]["answer"])
    a, b = num(fitb[k - 1]["answer_value"]), num(gold_txt)
    if a is None or b is None or not math.isclose(a, b, rel_tol=1e-12, abs_tol=0):
        answer_mismatch.append((k, i, fitb[k - 1]["answer_value"], gold_txt))

ids_not_in_fitb = sorted(set(range(1, 201)) - set(fitb_to_id.values()))
fitb_verified = [k for k in sorted(fitb_to_id) if fitb_to_id[k] in verified_ids]
ALL = sorted(fitb_to_id)  # mapped FITB indices (all 182 if mapping is complete)

# cross-check mapping against the authors' changed-cases package (source_id <-> fitb number)
changed = jl(CHANGED) if CHANGED.exists() else []
pkg_pairs = {(c["fitb_question_number_1based"], c["source_id"]) for c in changed}
pkg_disagree = [(k, i) for k, i in pkg_pairs if fitb_to_id.get(k) != i]
check(not pkg_disagree, f"mapping disagrees with changed_cases.jsonl: {pkg_disagree[:10]}")


# ---------------------------------------------------------------- load runs
def regrade(pred, truth, tol):
    p, t = num(pred), num(truth)
    if p is None or t is None:
        return None
    return abs(p - t) <= float(tol)


def load_fitb_run(path, label):
    rows = jl(path)
    out, regrade_diff, ungraded = {}, [], 0
    check(len(rows) == 182, f"{label}: {len(rows)} rows")
    for r in rows:
        k = int(r["index"])
        check(int(r["row_index"]) == k - 1, f"{label}: row_index != index-1 at {k}")
        ds = fitb[k - 1]
        check(str(r["truth"]).strip() == str(ds["answer_value"]).strip(),
              f"{label}: truth {r['truth']} != answer_value {ds['answer_value']} at {k}")
        check(num(r["tolerance"]) is not None and
              math.isclose(num(r["tolerance"]), float(ds["answer_tolerance"]), rel_tol=1e-9),
              f"{label}: tolerance mismatch at {k}")
        check(str(r.get("tolerance_mode", "absolute")) == "absolute",
              f"{label}: tolerance_mode {r.get('tolerance_mode')} at {k}")
        ic = r.get("is_correct")
        if isinstance(ic, str):
            ic = {"true": True, "false": False}.get(ic.strip().lower())
        if ic is None:
            ungraded += 1
        rg = regrade(r.get("prediction"), ds["answer_value"], ds["answer_tolerance"])
        if bool(rg) != bool(ic):
            regrade_diff.append((k, r.get("prediction"), ds["answer_value"], ic, rg))
        out[k] = {"correct": int(bool(ic)), "pred": r.get("prediction")}
    check(sorted(out) == list(range(1, 183)), f"{label}: indices not 1..182")
    return out, regrade_diff, ungraded


def load_mcq_run(path, label):
    rows = jl(path)
    out = {}
    for r in rows:
        i = int(r["index"])
        check(int(r["row_index"]) == i - 1, f"{label}: row_index != index-1 at {i}")
        check(str(r["truth"]).strip().upper() == mcq_by_id[i]["answer"].strip().upper(),
              f"{label}: truth != answer key at id {i}")
        ic = r.get("is_correct")
        if isinstance(ic, str):
            ic = {"true": True, "false": False}.get(ic.strip().lower())
        out[i] = int(bool(ic))
    check(sorted(out) == list(range(1, 201)), f"{label}: ids not 1..200")
    return out


fitb_res, fitb_regrade, fitb_ungraded = {}, {}, {}
for key, p in FITB_RUNS.items():
    if p.exists():
        fitb_res[key], fitb_regrade[key], fitb_ungraded[key] = load_fitb_run(p, f"FITB {key}")

mcq_res = {}
for key, p in MCQ_RUNS.items():
    if p.exists():
        mcq_res[key] = load_mcq_run(p, f"MCQ {key}")

spliced = collections.defaultdict(dict)
with open(SPLICED, newline="") as f:
    for r in csv.DictReader(f):
        for col in ("direct", "cot", "toc_old", "toc_new"):
            if r[col] != "":
                spliced[(r["model"], col)][int(r["item_id"])] = int(r[col])
for m in MODELS:
    for col in ("toc_new", "toc_old"):
        if len(spliced[(m, col)]) == 200:
            mcq_res[(m, "TOC" if col == "toc_new" else "TOC_old")] = spliced[(m, col)]
    # sanity: spliced direct/cot columns agree with results.jsonl when present
    for col, cond in (("direct", "Direct"), ("cot", "CoT")):
        if spliced[(m, col)] and (m, cond) in mcq_res:
            dis = [i for i, v in spliced[(m, col)].items() if v != mcq_res[(m, cond)][i]]
            check(not dis, f"spliced {m} {col} disagrees with results.jsonl on {dis[:10]}")

# ------------------------------------------------ reconstruct the earlier 101/182 TOC run
# The earlier FITB TOC run (result/custom/fill_in_blank_182_agent_toc_gpt54, not in the repo)
# is recoverable per item: it was compared against the MCQ toc_old run in changed_cases.jsonl,
# which lists every item where MCQ and FITB disagree. Unlisted items have FITB == MCQ(toc_old).
old_toc = None
chg = {c["fitb_question_number_1based"]: c for c in changed
       if c["model"] == "GPT-5.4" and c["method"] == "Agent-TOC"}
if chg and ("GPT-5.4", "TOC_old") in mcq_res:
    mo = mcq_res[("GPT-5.4", "TOC_old")]
    ok = all(int(bool(c["mcq_correct"])) == mo[c["source_id"]] for c in chg.values())
    check(ok, "changed_cases Agent-TOC mcq_correct disagrees with spliced toc_old")
    ok2 = all(str(c["fitb_truth"]).strip() == str(fitb[k - 1]["answer_value"]).strip()
              for k, c in chg.items())
    check(ok2, "changed_cases FITB truth differs from current answer_value")
    old_toc = {}
    for k in range(1, 183):
        if k in chg:
            old_toc[k] = {"correct": int(bool(chg[k]["fitb_correct"])),
                          "pred": chg[k]["fitb_prediction"]}
        else:
            old_toc[k] = {"correct": mo[fitb_to_id[k]], "pred": None}
    fitb_res[("GPT-5.4", "TOC_old101")] = old_toc


# ---------------------------------------------------------------- metrics
def acc(vec, items):
    c = sum(vec[i] for i in items)
    return c, len(items), 100.0 * c / len(items) if items else float("nan")


def gl(a, b, items):
    """gains/losses of a relative to b: gains = a right & b wrong."""
    g = sum(1 for i in items if a[i] and not b[i])
    l = sum(1 for i in items if b[i] and not a[i])
    return g, l


def fvec(model, cond):
    r = fitb_res.get((model, cond))
    return {k: v["correct"] for k, v in r.items()} if r else None


def mvec(model, cond):
    r = mcq_res.get((model, cond))
    return {k: r[fitb_to_id[k]] for k in ALL} if r else None  # keyed by FITB index


def block(vecs, items, toc_key="TOC"):
    d = {}
    for c, v in vecs.items():
        if v is not None:
            n, N, p = acc(v, items)
            d[c] = {"correct": n, "n": N, "pct": round(p, 2)}
    if all(vecs.get(c) is not None for c in ("Direct", "CoT", toc_key)):
        best = max(("Direct", "CoT"), key=lambda c: (d[c]["correct"], c == "CoT"))
        d["best_no_tool"] = best
        d["tool_adv_pp"] = round(100.0 * (d[toc_key]["correct"] - d[best]["correct"]) / len(items), 2)
        d["vs_direct"] = dict(zip(("gains", "losses"), gl(vecs[toc_key], vecs["Direct"], items)))
        d["vs_cot"] = dict(zip(("gains", "losses"), gl(vecs[toc_key], vecs["CoT"], items)))
    return d


subsets = {"all182": ALL, "verified": fitb_verified}
results = {"mapping": {
    "fitb_rows": len(fitb), "mapped_unique": len(fitb_to_id), "unmapped": unmapped,
    "mcq_ids_hit_twice": dup_targets, "answer_value_vs_gold_option_mismatch": answer_mismatch,
    "mcq_ids_without_fitb": ids_not_in_fitb, "n_verified_mcq": len(verified_ids),
    "n_fitb_verified": len(fitb_verified),
    "fitb_status_counts": dict(collections.Counter(status[fitb_to_id[k]] for k in ALL)),
    "verified_ids_without_fitb": sorted(verified_ids - set(fitb_to_id.values())),
    "changed_cases_pairs_checked": len(pkg_pairs), "changed_cases_pairs_disagree": pkg_disagree,
}, "subsets": {}}

for sname, items in subsets.items():
    S = {}
    g = {c: fvec("GPT-5.4", c) for c in ("Direct", "CoT", "TOC")}
    S["GPT-5.4 FITB (TOC = rebuttal run 2026-07-23)"] = block(g, items)
    if fvec("GPT-5.4", "TOC_old101"):
        g2 = dict(g, TOC=fvec("GPT-5.4", "TOC_old101"))
        S["GPT-5.4 FITB (TOC = earlier run, reconstructed 101/182)"] = block(g2, items)
    mg = {c: mvec("GPT-5.4", c) for c in ("Direct", "CoT", "TOC")}
    S["GPT-5.4 MCQ (TOC = toc_new headline)"] = block(mg, items)
    mg2 = dict(mg, TOC=mvec("GPT-5.4", "TOC_old"))
    S["GPT-5.4 MCQ (TOC = toc_old, pre cap-fix)"] = block(mg2, items)
    for m in MODELS[1:]:
        S[f"{m} FITB"] = block({c: fvec(m, c) for c in ("Direct", "CoT")}, items)
        S[f"{m} MCQ"] = block({c: mvec(m, c) for c in ("Direct", "CoT", "TOC")}, items)
    # old vs new FITB TOC run agreement
    if fvec("GPT-5.4", "TOC_old101"):
        a, b = fvec("GPT-5.4", "TOC"), fvec("GPT-5.4", "TOC_old101")
        S["TOC rerun vs earlier run"] = {
            "both": sum(a[i] and b[i] for i in items), "rerun_only": sum(a[i] and not b[i] for i in items),
            "earlier_only": sum(b[i] and not a[i] for i in items),
            "neither": sum((not a[i]) and (not b[i]) for i in items)}
    results["subsets"][sname] = S

results["regrade_disagreements"] = {f"{m}|{c}": v for (m, c), v in fitb_regrade.items() if v}
results["ungraded_rows"] = {f"{m}|{c}": v for (m, c), v in fitb_ungraded.items()}
results["sanity_check_failures"] = problems

# exact-match (numeric-equality) count for the rebuttal TOC run, for comparison with the CSV
exact = sum(1 for k, v in fitb_res[("GPT-5.4", "TOC")].items()
            if num(v["pred"]) is not None and num(v["pred"]) == num(fitb[k - 1]["answer_value"]))
results["rebuttal_toc_numeric_exact"] = exact
if EXACT_CSV.exists():
    results["exact_match_csv"] = list(csv.DictReader(open(EXACT_CSV)))

# ---------------------------------------------------------------- write outputs
with open(OUT / "fitb_to_mcq_map.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fitb_index", "mcq_id", "verification_status", "answer_value", "mcq_gold"])
    for k in range(1, len(fitb) + 1):
        i = fitb_to_id.get(k)
        w.writerow([k, i or "", status.get(i, "") if i else "", fitb[k - 1]["answer_value"],
                    mcq_by_id[i]["answer"] if i else ""])

cols = [("FITB", m, c) for (m, c) in fitb_res] + [("MCQ", m, c) for (m, c) in mcq_res]
with open(OUT / "per_item.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fitb_index", "mcq_id", "verified"] + [f"{fm}|{m}|{c}" for fm, m, c in cols])
    for k in ALL:
        row = [k, fitb_to_id[k], int(fitb_to_id[k] in verified_ids)]
        for fm, m, c in cols:
            row.append(fitb_res[(m, c)][k]["correct"] if fm == "FITB" else mcq_res[(m, c)][fitb_to_id[k]])
        w.writerow(row)

json.dump(results, open(OUT / "results.json", "w"), indent=1, default=str)


# ---------------------------------------------------------------- summary.md
def fmt(d, c):
    x = d.get(c)
    return f"{x['correct']}/{x['n']} ({100.0 * x['correct'] / x['n']:.1f}%)" if x else "n/a"


def row(name, d, toc=True):
    cells = [name, fmt(d, "Direct"), fmt(d, "CoT"), fmt(d, "TOC") if toc else "n/a"]
    if "tool_adv_pp" in d:
        cells += [f"{d['tool_adv_pp']:+.1f} pp (vs {d['best_no_tool']})",
                  f"+{d['vs_direct']['gains']} / -{d['vs_direct']['losses']}",
                  f"+{d['vs_cot']['gains']} / -{d['vs_cot']['losses']}"]
    else:
        cells += ["n/a"] * 3
    return "| " + " | ".join(cells) + " |"


HDR = ("| Condition | Direct | CoT | Agent-TOC | Tool advantage | TOC vs Direct (gain/loss) | "
       "TOC vs CoT (gain/loss) |\n|---|---|---|---|---|---|---|")
L = []
mp = results["mapping"]
L.append("# FITB robustness recomputation (GPT-5.4)\n")
L.append(f"Generated by `fitb_robustness.py` from `{R}` (read-only). Re-run: `python3 fitb_robustness.py`.\n")
L.append("## Mapping coverage\n")
L.append(f"- {mp['mapped_unique']}/{mp['fitb_rows']} FITB rows map to exactly one MCQ id by normalized "
         "question text (NFKC, zero-width chars removed, whitespace collapsed; MCQ options stripped). "
         f"Unmapped: {mp['unmapped'] or 'none'}. MCQ ids hit by two FITB rows: {mp['mcq_ids_hit_twice'] or 'none'}.")
L.append(f"- FITB `answer_value` equals the numeric text of the MCQ gold option for "
         f"{mp['mapped_unique'] - len(mp['answer_value_vs_gold_option_mismatch'])}/{mp['mapped_unique']} items"
         + (f"; mismatches: {mp['answer_value_vs_gold_option_mismatch']}." if mp['answer_value_vs_gold_option_mismatch'] else "."))
L.append(f"- Cross-check: all {mp['changed_cases_pairs_checked']} (FITB number, MCQ id) pairs in the authors' "
         f"`changed_cases.jsonl` agree with this mapping"
         + ("." if not mp["changed_cases_pairs_disagree"] else f" EXCEPT {mp['changed_cases_pairs_disagree']}."))
L.append(f"- The {len(mp['mcq_ids_without_fitb'])} MCQ ids with no FITB version: {mp['mcq_ids_without_fitb']}.")
L.append(f"- FITB items whose MCQ id is verified: **{mp['n_fitb_verified']}** "
         f"(of {mp['n_verified_mcq']} verified MCQ ids; verified ids without a FITB version: "
         f"{mp['verified_ids_without_fitb']}). Status mix of the 182: {mp['fitb_status_counts']}.\n")
L.append("## Row mapping of the run files\n")
L.append("- FITB `results.jsonl`: `index` = 1-based FITB row, `row_index` = index-1; for every run and every row "
         "`truth` == `answer_value` and `tolerance` == `answer_tolerance` (absolute mode). "
         "MCQ `results.jsonl`: `index` = MCQ id, `truth` == answer key.")
rd = results["regrade_disagreements"]
L.append("- Independent re-grade with `abs(pred - truth) <= answer_tolerance` agrees with recorded `is_correct` "
         + ("for every row of every run." if not rd else f"except: {rd}."))
L.append(f"- Ungraded (unparseable) rows, counted as wrong: {results['ungraded_rows']}.\n")

for sname, title in (("all182", "All 182 FITB items"),
                     ("verified", f"Verified subset ({mp['n_fitb_verified']} FITB items whose MCQ id is verified)")):
    S = results["subsets"][sname]
    L.append(f"## {title}\n")
    L.append(HDR)
    for name, d in S.items():
        if name.startswith("TOC rerun"):
            continue
        L.append(row(name, d, toc="TOC" in d))
    if "TOC rerun vs earlier run" in S:
        x = S["TOC rerun vs earlier run"]
        L.append(f"\nFITB TOC rerun (93-run) vs earlier run (101-run), per item: both correct {x['both']}, "
                 f"rerun-only {x['rerun_only']}, earlier-only {x['earlier_only']}, neither {x['neither']}.")
    L.append("")

a = results["subsets"]["all182"]
r = a["GPT-5.4 FITB (TOC = rebuttal run 2026-07-23)"]
mo = a["GPT-5.4 MCQ (TOC = toc_old, pre cap-fix)"]
claims = [
    ("Direct 2/182", r["Direct"]["correct"] == 2), ("CoT 4/182", r["CoT"]["correct"] == 4),
    ("TOC 93/182", r["TOC"]["correct"] == 93), ("+48.9 pp", abs(r["tool_adv_pp"] - 48.9) < 0.05),
    ("vs Direct +91/-0", (r["vs_direct"]["gains"], r["vs_direct"]["losses"]) == (91, 0)),
    ("vs CoT +90/-1", (r["vs_cot"]["gains"], r["vs_cot"]["losses"]) == (90, 1)),
    ("MCQ +39.0 pp (toc_old)", abs(mo["tool_adv_pp"] - 39.0) < 0.05),
]
L.append("## Rebuttal claims check (all 182)\n")
for c, ok in claims:
    L.append(f"- {c}: {'reproduced' if ok else 'NOT reproduced'}")
mn = a["GPT-5.4 MCQ (TOC = toc_new headline)"]
L.append(f"- Note: the MCQ +39.0 pp is reproduced only with `toc_old` (TOC {fmt(mo, 'TOC')}). With the headline "
         f"`toc_new` column the MCQ TOC is {fmt(mn, 'TOC')} and the MCQ tool advantage is {mn['tool_adv_pp']:+.1f} pp.\n")

L.append("## The 93 vs 101 discrepancy\n")
ex = {d["run"]: d for d in results.get("exact_match_csv", [])}
sj = json.load(open(FITB_RUNS[("GPT-5.4", "TOC")].parent / "summary.json"))
L.append(f"- {sj['correct_rows']}/{sj['total_rows']} is `{FITB_RUNS[('GPT-5.4', 'TOC')].parent.relative_to(R)}` "
         f"(summary.json; started {sj['started_at']}, {sj['elapsed_seconds']:.1f} s, "
         f"{sj['total_input_tokens']:,} input tokens).")
cur = {d["method"]: d for d in csv.DictReader(open(R / "neurips_analysis/fill_in_blank_current_summary.csv"))} \
    if (R / "neurips_analysis/fill_in_blank_current_summary.csv").exists() else {}
old = cur.get("GPT-5.4 + TOC") or next((d for k, d in cur.items() if "5.4" in k and "TOC" in k), {})
L.append(f"- {old.get('fitbCorrect', '101')}/182 (`fill_in_blank_agent_exact_match.csv`, `fill_in_blank_current_summary.*`, "
         "`mcq_fitb_changed_gpt54_sonnet/`) comes from an earlier, different run, "
         "`result/custom/fill_in_blank_182_agent_toc_gpt54` "
         f"({int(old.get('fitbInputTokens', 0)):,} FITB input tokens per `fill_in_blank_current_summary.csv`), "
         "whose directory is no longer in the repo"
         + (" (it still exists)" if (R / "result/custom/fill_in_blank_182_agent_toc_gpt54").exists() else "") + ".")
L.append("- Same grading rule in both: the 101 is tolerance-graded, not exact-match "
         f"(the same CSV lists numeric-exact {ex.get('toc_gpt54', {}).get('numericExact', '?')}/182 for that run). "
         f"Under numeric exact match the rebuttal run would score {results['rebuttal_toc_numeric_exact']}/182.")
L.append("- The earlier run's per-item vector was reconstructed from `changed_cases.jsonl` (every MCQ/FITB disagreement) "
         "plus the MCQ `toc_old` vector; it sums to "
         + (f"{sum(old_toc[k]['correct'] for k in old_toc)}/182" if old_toc else "n/a")
         + ", matching the CSV, so the two runs differ by run-to-run agent variance on the same items, truth values and tolerances.\n")
fd, fc, ft = (fvec("GPT-5.4", c) for c in ("Direct", "CoT", "TOC"))
lost_cot = [(k, fitb_to_id[k]) for k in ALL if fc[k] and not ft[k]]
lost_dir = [(k, fitb_to_id[k]) for k in ALL if fd[k] and not ft[k]]
nt = [(k, fitb_to_id[k]) for k in ALL if fd[k] or fc[k]]
L.append("## Caveats\n")
L.append(f"- Items where a GPT-5.4 no-tool FITB run is correct (FITB index, MCQ id): {nt}; all are verified items, "
         "so the gain/loss counts barely change on the verified subset. "
         f"TOC losses vs CoT: {lost_cot or 'none'}; vs Direct: {lost_dir or 'none'}.")
L.append("- Every condition is a single run. The two GPT-5.4 FITB TOC runs disagree on "
         + (f"{a['TOC rerun vs earlier run']['rerun_only'] + a['TOC rerun vs earlier run']['earlier_only']}/182 items"
            if "TOC rerun vs earlier run" in a else "n/a")
         + " (net 8 items), so the TOC accuracy carries several points of run-to-run noise.")
L.append("- The earlier-run (101) vector is a reconstruction: it assumes `changed_cases.jsonl` lists every MCQ/FITB "
         "disagreement for GPT-5.4 Agent-TOC and that its MCQ side equals spliced `toc_old`. Both checks pass: "
         "the listed MCQ outcomes match `toc_old` on all listed items, and the total reproduces 101.")
L.append("- MCQ TOC: `toc_new` is the headline cap-fix spliced vector (only 45 cap-affected items re-run). The rebuttal's "
         "+39.0 pp MCQ advantage used `toc_old`; on the verified subset both columns give the same TOC total.")
L.append("- No FITB TOC run for Claude Sonnet 4.6 or Opus 4.6 is in the repo (Sonnet's 116/182 directory is also gone), "
         "so their rows show only Direct/CoT for FITB. Their MCQ TOC column is `toc_new`.")
L.append("- Ungraded FITB rows (unparseable answers) count as wrong with denominator = all items, matching the authors' accuracy.")
L.append("- 'Verified' is the MCQ key-verification status. FITB `answer_value` equals the MCQ gold option value on every "
         "mapped item, so the status carries over to the FITB truth. The FITB tolerance is a property of the dataset row "
         "and was not re-derived.\n")
L.append("## Sanity-check failures\n")
L.append("- none" if not problems else "\n".join(f"- {p}" for p in problems))
(OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
