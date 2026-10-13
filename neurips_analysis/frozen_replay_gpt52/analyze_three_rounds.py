"""Recompute the completed three-round frozen replay from immutable branches.

No API, simulator or runner imports. Writes only beside this script; preserves
the round-one analysis, all checkpoints, branch states and raw result files.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUN = REPO / "result/custom/frozen_replay_gpt52"
ROUNDS = (1, 2, 3)
SEED = 20261011
BOOTSTRAPS = 50_000


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_final(text):
    wrapped = re.search(r"<<<\s*(.+?)\s*>>>", text, re.DOTALL)
    if not wrapped:
        return None
    answer = wrapped.group(1).strip()
    return answer.upper() if re.fullmatch(r"[A-D]", answer, re.IGNORECASE) else None


def percentile_interval(values):
    values = sorted(values)
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def stats(items):
    n = len(items)
    full = sum(x["full_correct_count"] for x in items)
    toc = sum(x["toc_correct_count"] for x in items)
    return {"items": n, "repeats_per_condition": 3, "paired_continuations": 3 * n,
            "full_correct": full, "toc_correct": toc,
            "full_mean_accuracy_pct": 100 * full / (3 * n),
            "toc_mean_accuracy_pct": 100 * toc / (3 * n),
            "toc_minus_full_pp": 100 * sum(x["difference_sum"] for x in items) / (3 * n)}


def main():
    manifest_path = RUN / "checkpoints/manifest.json"
    experiment_path = RUN / "experiment.json"
    scenario_path = REPO / "human_evaluation/scenarios.json"
    manifest, experiment, scenarios = map(read, (manifest_path, experiment_path, scenario_path))
    assert manifest["status"] == "complete" and experiment["repeats"] == 3
    assert file_hash(manifest_path) == experiment["manifest_sha256"]
    source_hashes = {str(p.relative_to(REPO)): file_hash(p)
                     for p in (manifest_path, experiment_path, scenario_path)}
    dataset_path = REPO / "datasets/neurips_dataset_158_verified.jsonl"
    gold = {x["id"]: x["answer"] for x in map(json.loads, dataset_path.read_text().splitlines())}
    source_hashes[str(dataset_path.relative_to(REPO))] = file_hash(dataset_path)
    matcher_path = REPO / "neurips_analysis/frozen_replay_gpt52_round1/item_analysis.json"
    matcher = {x["item_id"]: x for x in read(matcher_path)} if matcher_path.exists() else {}
    if matcher:
        source_hashes[str(matcher_path.relative_to(REPO))] = file_hash(matcher_path)
    items, branches = [], []
    cost = 0.0
    for entry in sorted(manifest["eligible"], key=lambda x: x["item_id"]):
        iid = entry["item_id"]
        cp_path = RUN / "checkpoints" / entry["checkpoint"]
        cp = read(cp_path)
        assert cp["item_id"] == iid and cp["item"]["answer"] == gold[iid]
        assert digest(cp) == experiment["checkpoint_hashes"][str(iid)]
        snapshot = cp_path.parent / cp["snapshot"]
        snapshot_hashes = {}
        for path in sorted(snapshot.rglob("*")):
            assert not path.is_symlink(), path
            if path.is_file():
                snapshot_hashes[str(path.relative_to(snapshot))] = file_hash(path)
        assert snapshot_hashes == cp["snapshot_hashes"]
        report = snapshot / cp["report"]["path"]
        assert file_hash(report) == cp["report"]["sha256"]
        assert len(report.read_text()) == cp["report"]["chars"] == entry["report_chars"]
        assert entry["inline_truncated"] == (entry["report_chars"] > 100_000)
        group = [s["id"] for s in scenarios if s["start"] <= iid <= s["end"]]
        assert len(group) == 1
        row = {"item_id": iid, "scenario_id": group[0], "report_chars": entry["report_chars"],
               "inline_truncated": entry["inline_truncated"], "remaining_steps": cp["remaining_steps"],
               "truth": gold[iid], "rounds": []}
        if matcher:
            row["keyed_value_in_checkpoint_report"] = matcher[iid]["gold_in_checkpoint_report"]
            row["keyed_value_in_any_snapshot_file"] = bool(matcher[iid]["gold_matches_in_snapshot"])
        for repeat in ROUNDS:
            pair = {"repeat": repeat}
            for arm in ("full", "toc"):
                folder = RUN / "runs" / f"repeat_{repeat:02d}" / f"item_{iid:05d}" / arm
                state_path, result_path = folder / "state.json", folder / "result.json"
                state, result = read(state_path), read(result_path)
                fingerprint = digest({"checkpoint": cp, "condition": arm, "config": experiment["config"],
                                      "system": experiment["system_prompt"], "tools": experiment["tool_specs"]})
                assert state["phase"] == "done"
                assert state["fingerprint"] == fingerprint == result["checkpoint_fingerprint"]
                assert state["condition"] == result["condition"] == arm
                assert result["item_id"] == iid and result["truth"] == gold[iid]
                assert state["calls"] == result["calls"] == len(state["usage"])
                assert state["calls"] <= result["remaining_steps"] == cp["remaining_steps"]
                assert state["status"] == result["status"]
                in_tokens = sum(u.get("prompt_tokens", 0) for u in state["usage"])
                out_tokens = sum(u.get("completion_tokens", 0) for u in state["usage"])
                assert in_tokens == result["input_tokens"] and out_tokens == result["output_tokens"]
                branch_cost = (in_tokens * 1.75 + out_tokens * 14) / 1_000_000
                assert math.isclose(branch_cost, state["estimated_cost_usd"], abs_tol=1e-9)
                assert math.isclose(branch_cost, result["estimated_cost_usd"], abs_tol=1e-9)
                cost += branch_cost
                answer_path = folder / "final_answer.txt"
                prediction = parse_final(answer_path.read_text()) if answer_path.exists() else None
                correct = prediction == gold[iid]
                assert result["prediction"] == prediction
                assert result["is_correct"] is (None if prediction is None else correct)
                for event in state["events"]:
                    if event["result"].get("ok"):
                        assert event["tool"] in {"list_file", "read_file", "write_file"}
                        if event["tool"] == "write_file":
                            assert event["args"]["path"] in {"final_answer.txt", "./final_answer.txt"}
                pair[arm] = {"prediction": prediction, "correct": correct, "status": result["status"],
                             "calls": result["calls"], "input_tokens": in_tokens, "output_tokens": out_tokens}
                branches.append({"item_id": iid, "repeat": repeat, "condition": arm,
                                 **pair[arm], "estimated_cost_usd": branch_cost})
                for path in (state_path, result_path):
                    source_hashes[str(path.relative_to(REPO))] = file_hash(path)
                if answer_path.exists():
                    source_hashes[str(answer_path.relative_to(REPO))] = file_hash(answer_path)
            pair["difference"] = int(pair["toc"]["correct"]) - int(pair["full"]["correct"])
            row["rounds"].append(pair)
        row["full_correct_count"] = sum(x["full"]["correct"] for x in row["rounds"])
        row["toc_correct_count"] = sum(x["toc"]["correct"] for x in row["rounds"])
        row["difference_sum"] = sum(x["difference"] for x in row["rounds"])
        row["mean_paired_difference"] = row["difference_sum"] / 3
        items.append(row)
    assert len(items) == 131 and len(branches) == 786
    observed = stats(items)
    diffs = [row["difference_sum"] for row in items]
    rng = random.Random(SEED)
    item_ci = percentile_interval([100 * sum(rng.choices(diffs, k=len(items))) / (3 * len(items))
                                   for _ in range(BOOTSTRAPS)])
    grouped = defaultdict(list)
    for row in items:
        grouped[row["scenario_id"]].append(row)
    assert len(grouped) == 19
    totals = [(len(group), sum(row["difference_sum"] for row in group)) for _, group in sorted(grouped.items())]
    rng = random.Random(SEED)
    sampled_stats = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choices(totals, k=len(totals))
        sampled_stats.append(100 * sum(d for _, d in sampled) / (3 * sum(n for n, _ in sampled)))
    cluster_ci = percentile_interval(sampled_stats)
    loo = [{"omitted_setup": group, **stats([row for row in items if row["scenario_id"] != group])}
           for group in sorted(grouped)]
    by_repeat = []
    for repeat in ROUNDS:
        group = [row["rounds"][repeat - 1] for row in items]
        full, toc = (sum(x[arm]["correct"] for x in group) for arm in ("full", "toc"))
        by_repeat.append({"repeat": repeat, "items": len(items), "full_correct": full,
                          "toc_correct": toc, "full_accuracy_pct": 100 * full / len(items),
                          "toc_accuracy_pct": 100 * toc / len(items),
                          "toc_minus_full_pp": 100 * (toc - full) / len(items),
                          "full_no_commit": sum(x["full"]["prediction"] is None for x in group),
                          "toc_no_commit": sum(x["toc"]["prediction"] is None for x in group)})
    strata = {"at_most_100000_chars": stats([x for x in items if not x["inline_truncated"]]),
              "over_100000_chars": stats([x for x in items if x["inline_truncated"]])}
    key_strata = {"matched_keyed_value": stats([x for x in items if x["keyed_value_in_checkpoint_report"]]),
                  "no_matched_keyed_value": stats([x for x in items if not x["keyed_value_in_checkpoint_report"]])} if matcher else None
    summary = {
        "scope": "Three paired continuations per condition on each of 131 fixed eligible source checkpoints; uncertainty resamples 131 items, never 393 independent pairs.",
        "seed": SEED, "bootstrap_replicates": BOOTSTRAPS,
        "overall": {**observed, "item_bootstrap_95ci_pp": item_ci, "setup_bootstrap_95ci_pp": cluster_ci},
        "by_repeat": by_repeat, "setup_count": len(grouped),
        "by_setup": [{"scenario_id": group, **stats(rows)} for group, rows in sorted(grouped.items())],
        "leave_one_setup_out": loo,
        "leave_one_setup_out_range_pp": [min(x["toc_minus_full_pp"] for x in loo), max(x["toc_minus_full_pp"] for x in loo)],
        "cap_strata": strata, "keyed_value_match_strata": key_strata,
        "methods": {
            "primary": "For each item, average its three binary outcomes within each condition; subtract Full from TOC, then average these 131 paired differences.",
            "item_bootstrap": "Resample 131 complete items with replacement, retaining all three rounds and both arms; 95% percentile interval.",
            "setup_bootstrap": "Sensitivity: resample 19 author setups with replacement, retaining every item, all three rounds and both arms in each sampled setup. Compute sampled item-weighted mean difference (ratio of sum differences to sampled item count).",
            "setup_map": "human_evaluation/scenarios.json original item-ID ranges; all setups except S17 are represented. No older 21-stem grouping is used.",
            "cap": "Actual frozen report character count from each checkpoint/manifest, with exactly 100000 included in the untruncated stratum.",
            "keyed_value_layer": "Reuse frozen_replay_gpt52_round1/item_analysis.json. Existing mechanism matcher make_spec/clean_text/match_text applies option-precision numeric matching, quantity sign handling, and specific-conductance context restrictions. A numeric occurrence is not semantic verification that the requested quantity was correctly computed.",
            "grading": "Read final_answer.txt independently; parse the first <<<...>>> wrapper containing exactly one A-D letter (case-insensitive), compare to verified dataset key. Missing/unparseable answer is incorrect for accuracy; compare against stored prediction/is_correct fields.",
            "inference": "No pooled McNemar test is performed. These intervals describe uncertainty over items/setups conditional on the fixed source checkpoints and three observed repeats; they do not cover source-trajectory selection or reconstruction uncertainty.",
        },
        "audit": {
            "checked_checkpoints_and_snapshot_hashes": len(items), "checked_completed_branch_states": len(branches),
            "regraded_final_answers": len(branches), "grading_disagreements": 0,
            "manifest_and_checkpoint_fingerprints_match": True,
            "branch_fingerprints_match_experiment_configuration": True,
            "no_successful_simulation_or_nonfinal_write_during_replay": True,
            "current_runner_matches_recorded_sha256": file_hash(REPO / "scripts/run_frozen_replay.py") == experiment["config"]["runner_sha256"],
            "status_counts": {arm: dict(Counter(r["status"] for r in branches if r["condition"] == arm)) for arm in ("full", "toc")},
            "no_commit_branches": [r for r in branches if r["prediction"] is None],
            "estimated_cost_usd_from_usage": cost,
        },
        "source_sha256": source_hashes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "three_round_items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    (OUT / "three_round_statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "GPT-5.2 frozen-computation replay: three rounds",
        "131 fixed eligible checkpoints, 19 author setups, 3 repeats per condition (786 branches).",
        f"Full: {observed['full_correct']}/393 ({observed['full_mean_accuracy_pct']:.2f}%); TOC: {observed['toc_correct']}/393 ({observed['toc_mean_accuracy_pct']:.2f}%).",
        f"Item-averaged TOC - Full: {observed['toc_minus_full_pp']:.2f} pp; 95% paired-item bootstrap CI [{item_ci[0]:.2f}, {item_ci[1]:.2f}] pp.",
        f"Setup-bootstrap sensitivity interval: [{cluster_ci[0]:.2f}, {cluster_ci[1]:.2f}] pp.",
        f"Leave-one-setup-out range: [{summary['leave_one_setup_out_range_pp'][0]:.2f}, {summary['leave_one_setup_out_range_pp'][1]:.2f}] pp.",
        "",
    ]
    for row in by_repeat:
        report.append(f"Round {row['repeat']}: Full {row['full_correct']}/131; TOC {row['toc_correct']}/131; difference {row['toc_minus_full_pp']:.2f} pp; TOC no-commit {row['toc_no_commit']}.")
    for label, row in strata.items():
        report.append(f"Report {label}: {row['items']} items; Full {row['full_correct']}/{row['paired_continuations']}; TOC {row['toc_correct']}/{row['paired_continuations']}; difference {row['toc_minus_full_pp']:.2f} pp.")
    report.extend(["", f"50,000 bootstrap draws, seed {SEED}; resampling retains all three rounds within each item.",
                   "The 393 paired continuations are not treated as independent observations. No pooled McNemar test.",
                   "Audit: every checkpoint/snapshot hash and branch fingerprint matched; all 786 final answers regraded with zero disagreements.",
                   "One TOC branch (round 2, item 150) reached its 13 remaining steps without commitment; scored incorrect.",
                   "Inference is conditional on the fixed source trajectories and eligible checkpoints.",
                   "See three_round_statistics.json for complete methods, strata, setup sensitivity and source hashes."])
    (OUT / "three_round_summary.txt").write_text("\n".join(report) + "\n")
    macro_lines = [
        "% Generated by neurips_analysis/frozen_replay_gpt52/analyze_three_rounds.py.",
        "% Three repeats per condition; inference resamples 131 items, preserving all repeats.",
        "% Accuracy macros are bare percentages (add \\%); differences are bare pp (add units).",
        "% Separate from the original trajectory/simulator-call totals in the manuscript.",
    ]

    def number(value):
        return f"{value:.1f}".replace("-", "$-$")

    def macro(name, value):
        macro_lines.append("\\newcommand{\\Replay%s}{%s}" % (name, value))

    for name, value in {"Items": len(items), "Runs": 3, "Branches": len(branches),
                        "Pairs": 3 * len(items), "Setups": len(grouped),
                        "FullCorrect": observed["full_correct"], "TOCCorrect": observed["toc_correct"],
                        "FullNoCommit": sum(r["prediction"] is None for r in branches if r["condition"] == "full"),
                        "TOCNoCommit": sum(r["prediction"] is None for r in branches if r["condition"] == "toc")}.items():
        macro(name, value)
    macro("FullAcc", number(observed["full_mean_accuracy_pct"]))
    macro("TOCAcc", number(observed["toc_mean_accuracy_pct"]))
    macro("Delta", number(observed["toc_minus_full_pp"]))
    macro("CI", "[%s, %s]" % tuple(map(number, item_ci)))
    macro("ClusterCI", "[%s, %s]" % tuple(map(number, cluster_ci)))
    macro("LOOMin", number(summary["leave_one_setup_out_range_pp"][0]))
    macro("LOOMax", number(summary["leave_one_setup_out_range_pp"][1]))
    macro_groups = [("Fit", strata["at_most_100000_chars"]), ("Long", strata["over_100000_chars"])]
    if key_strata:
        macro_groups.append(("Key", key_strata["matched_keyed_value"]))
    for name, group in macro_groups:
        macro(name + "N", group["items"])
        macro(name + "Pairs", group["paired_continuations"])
        macro(name + "FullCorrect", group["full_correct"])
        macro(name + "TOCCorrect", group["toc_correct"])
        macro(name + "FullAcc", number(group["full_mean_accuracy_pct"]))
        macro(name + "TOCAcc", number(group["toc_mean_accuracy_pct"]))
        macro(name + "Delta", number(group["toc_minus_full_pp"]))
    for name, group in zip(("One", "Two", "Three"), by_repeat):
        macro("Run" + name + "FullCorrect", group["full_correct"])
        macro("Run" + name + "TOCCorrect", group["toc_correct"])
        macro("Run" + name + "FullAcc", number(group["full_accuracy_pct"]))
        macro("Run" + name + "TOCAcc", number(group["toc_accuracy_pct"]))
        macro("Run" + name + "Delta", number(group["toc_minus_full_pp"]))
    (OUT / "results_macros_replay.tex").write_text("\n".join(macro_lines) + "\n")
    print(json.dumps({key: summary[key] for key in ("overall", "by_repeat", "cap_strata", "keyed_value_match_strata", "leave_one_setup_out_range_pp", "audit")}, indent=2))


if __name__ == "__main__":
    main()
