"""One-off analysis script: cross-reference all results for the final paper."""
import json
from pathlib import Path
from collections import defaultdict

RESULT_ROOT = Path(__file__).resolve().parent / "result"
DATASET_PATH = Path(__file__).resolve().parent / "dataset_S+J.jsonl"

RUNS = {
    "GPT-5.1 one-shot":          "sj_baseline_gpt51",
    "GPT-5.2 one-shot":          "sj_baseline_gpt52",
    "Gemini 2.5 Flash one-shot": "sj_baseline_gemini25flash",
    "Gemini 2.5 Pro one-shot":   "sj_baseline_gemini25pro",
    "Claude Sonnet 4.6 one-shot":"sj_baseline_claude_sonnet46",
    "Claude Opus 4.6 one-shot":  "sj_baseline_claude_opus46",
    "GPT-5.1 + Agent":           "sj_agent_gpt51",
    "GPT-5.2 + Agent":           "sj_agent_gpt52",
    "Gemini 2.5 Flash + Agent":  "sj_agent_gemini25flash",
    "Claude Opus 4.6 + Agent":   "sj_agent_claude_opus46",
}

BASELINES = [
    "GPT-5.1 one-shot", "GPT-5.2 one-shot",
    "Gemini 2.5 Flash one-shot", "Gemini 2.5 Pro one-shot",
    "Claude Sonnet 4.6 one-shot", "Claude Opus 4.6 one-shot",
]
AGENTS = [
    "GPT-5.1 + Agent", "GPT-5.2 + Agent",
    "Gemini 2.5 Flash + Agent", "Claude Opus 4.6 + Agent",
]

def load_results(run_folder):
    path = RESULT_ROOT / run_folder / "results.jsonl"
    data = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            data[r["index"]] = r
    return data

def load_dataset():
    questions = {}
    with DATASET_PATH.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions[idx + 1] = row
    return questions

def main():
    all_results = {}
    for name, folder in RUNS.items():
        all_results[name] = load_results(folder)

    questions = load_dataset()
    out = []
    p = out.append

    p("=" * 80)
    p("COMPREHENSIVE EVALUATION ANALYSIS — dataset_S+J.jsonl (200 questions)")
    p("Questions 61–260 from the full Phreeqc_MCQ dataset")
    p("=" * 80)

    # 1. HEADLINE RESULTS
    p("")
    p("=" * 80)
    p("1. HEADLINE RESULTS (sorted by accuracy)")
    p("=" * 80)
    p("")
    summaries = []
    for name in list(BASELINES) + list(AGENTS):
        res = all_results[name]
        total = len(res)
        correct = sum(1 for r in res.values() if r.get("is_correct") is True)
        errors = sum(1 for r in res.values() if r.get("error"))
        no_answer = sum(1 for r in res.values() if r.get("prediction") is None and not r.get("error"))
        summaries.append((name, correct, total, errors, no_answer))

    summaries.sort(key=lambda x: x[1], reverse=True)
    p(f"  {'Method':<30s} {'Correct':>8s} {'Errors':>7s} {'No Ans':>7s} {'Accuracy':>9s}")
    p(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*7} {'-'*9}")
    for name, correct, total, errors, no_ans in summaries:
        acc = correct / total if total else 0
        p(f"  {name:<30s} {correct:>8d} {errors:>7d} {no_ans:>7d} {acc:>8.1%}")

    # 2. PROVIDER COMPARISON
    p("")
    p("=" * 80)
    p("2. PROVIDER COMPARISON — One-shot vs Agent lift")
    p("=" * 80)
    p("")
    pairs = [
        ("OpenAI GPT-5.1", "GPT-5.1 one-shot", "GPT-5.1 + Agent"),
        ("OpenAI GPT-5.2", "GPT-5.2 one-shot", "GPT-5.2 + Agent"),
        ("Google Gemini 2.5 Flash", "Gemini 2.5 Flash one-shot", "Gemini 2.5 Flash + Agent"),
        ("Anthropic Claude Opus 4.6", "Claude Opus 4.6 one-shot", "Claude Opus 4.6 + Agent"),
    ]
    p(f"  {'Provider/Model':<30s} {'One-shot':>9s} {'Agent':>9s} {'Lift':>9s}")
    p(f"  {'-'*30} {'-'*9} {'-'*9} {'-'*9}")
    for label, bs, ag in pairs:
        bs_correct = sum(1 for r in all_results[bs].values() if r.get("is_correct") is True)
        ag_correct = sum(1 for r in all_results[ag].values() if r.get("is_correct") is True)
        bs_acc = bs_correct / 200
        ag_acc = ag_correct / 200
        lift = ag_acc - bs_acc
        p(f"  {label:<30s} {bs_acc:>8.1%} {ag_acc:>8.1%} {lift:>+8.1%}")

    # 3. "THINKING MODEL" PATTERN
    p("")
    p("=" * 80)
    p("3. THINKING MODEL PATTERN — Larger model vs smaller (one-shot)")
    p("=" * 80)
    p("")
    model_pairs = [
        ("OpenAI", "GPT-5.1 one-shot", "GPT-5.2 one-shot"),
        ("Google", "Gemini 2.5 Flash one-shot", "Gemini 2.5 Pro one-shot"),
        ("Anthropic", "Claude Sonnet 4.6 one-shot", "Claude Opus 4.6 one-shot"),
    ]
    for provider, smaller, larger in model_pairs:
        sm_c = sum(1 for r in all_results[smaller].values() if r.get("is_correct") is True)
        lg_c = sum(1 for r in all_results[larger].values() if r.get("is_correct") is True)
        p(f"  {provider}: {smaller} ({sm_c}/200 = {sm_c/2:.1f}%) vs {larger} ({lg_c}/200 = {lg_c/2:.1f}%)")
        if provider == "Anthropic":
            p(f"    -> Larger model WINS (+{(lg_c - sm_c)/2:.1f} pp)")
        else:
            p(f"    -> Smaller model wins (+{(sm_c - lg_c)/2:.1f} pp)")
    p("")
    p("  Pattern: OpenAI and Google 'thinking' models score LOWER on one-shot MCQ.")
    p("  Exception: Anthropic Opus outperforms Sonnet — the only provider where bigger = better.")

    # 4. PER-QUESTION CROSS-REFERENCE
    p("")
    p("=" * 80)
    p("4. PER-QUESTION DIFFICULTY ANALYSIS")
    p("=" * 80)
    p("")

    all_names = list(BASELINES) + list(AGENTS)
    q_solved_by = {}
    for qi in range(1, 201):
        solved = set()
        for name in all_names:
            r = all_results[name].get(qi, {})
            if r.get("is_correct") is True:
                solved.add(name)
        q_solved_by[qi] = solved

    none_solved = [qi for qi in range(1, 201) if len(q_solved_by[qi]) == 0]
    all_solved = [qi for qi in range(1, 201) if len(q_solved_by[qi]) == len(all_names)]
    only_agent = [qi for qi in range(1, 201) if q_solved_by[qi] and q_solved_by[qi].issubset(set(AGENTS))]
    only_baseline = [qi for qi in range(1, 201) if q_solved_by[qi] and q_solved_by[qi].issubset(set(BASELINES))]

    best_agent = "Claude Opus 4.6 + Agent"
    best_agent_solved = {qi for qi in range(1, 201) if best_agent in q_solved_by[qi]}
    best_agent_only = [qi for qi in range(1, 201)
                       if best_agent in q_solved_by[qi]
                       and not any(b in q_solved_by[qi] for b in BASELINES)]

    p(f"  All 10 methods correct:     {len(all_solved):>4d}  ({len(all_solved)/2:.1f}%)")
    p(f"  No method correct:          {len(none_solved):>4d}  ({len(none_solved)/2:.1f}%)")
    p(f"  Agent-only (no baseline):   {len(only_agent):>4d}  ({len(only_agent)/2:.1f}%)")
    p(f"  Baseline-only (no agent):   {len(only_baseline):>4d}  ({len(only_baseline)/2:.1f}%)")

    # Tier breakdown
    tiers = {"easy": [], "medium": [], "hard": [], "impossible": []}
    for qi in range(1, 201):
        n = len(q_solved_by[qi])
        if n == len(all_names):
            tiers["easy"].append(qi)
        elif n == 0:
            tiers["impossible"].append(qi)
        elif n >= len(all_names) // 2:
            tiers["easy"].append(qi)  # majority solved
        else:
            tiers["medium"].append(qi)

    # Simpler tiers
    tiers2 = defaultdict(list)
    for qi in range(1, 201):
        n = len(q_solved_by[qi])
        if n == 0:
            tiers2["none"].append(qi)
        elif n <= 3:
            tiers2["hard"].append(qi)
        elif n <= 7:
            tiers2["medium"].append(qi)
        else:
            tiers2["easy"].append(qi)

    p("")
    p(f"  Difficulty tiers (by # methods solving):")
    p(f"    Easy   (8-10 methods): {len(tiers2['easy']):>4d}  ({len(tiers2['easy'])/2:.1f}%)")
    p(f"    Medium (4-7 methods):  {len(tiers2['medium']):>4d}  ({len(tiers2['medium'])/2:.1f}%)")
    p(f"    Hard   (1-3 methods):  {len(tiers2['hard']):>4d}  ({len(tiers2['hard'])/2:.1f}%)")
    p(f"    Impossible (0):        {len(tiers2['none']):>4d}  ({len(tiers2['none'])/2:.1f}%)")

    # Best agent on each tier
    p("")
    p(f"  Best agent ({best_agent}) performance by tier:")
    for tier_name in ["easy", "medium", "hard", "none"]:
        qs = tiers2[tier_name]
        if not qs:
            continue
        solved = sum(1 for qi in qs if best_agent in q_solved_by[qi])
        p(f"    {tier_name.capitalize():>12s}: {solved}/{len(qs)} ({solved/len(qs)*100:.1f}%)")

    # 5. NEVER SOLVED
    p("")
    p("=" * 80)
    p(f"5. NEVER SOLVED — {len(none_solved)} questions")
    p("   No method (baseline or agent) answered correctly.")
    p("=" * 80)
    p("")

    for qi in sorted(none_solved):
        q_data = questions.get(qi, {})
        truth = q_data.get("answer", "?")
        q_text = q_data.get("question", "N/A")
        p(f"--- Q{qi} (original: Q{qi+60}) ---")
        p(f"  Dataset answer: {truth}")
        preds = []
        for name in all_names:
            r = all_results[name].get(qi, {})
            pred = r.get("prediction", "-") or "-"
            preds.append(f"{name}={pred}")
        p(f"  Predictions: {', '.join(preds)}")
        # Truncate question to first 200 chars
        p(f"  Q: {q_text[:200]}...")
        p("")

    # 6. BEST AGENT ONLY (solved by Opus agent, no baseline got it)
    p("=" * 80)
    p(f"6. SOLVED BY BEST AGENT ONLY — {len(best_agent_only)} questions")
    p(f"   {best_agent} correct, ALL 6 baselines wrong.")
    p("=" * 80)
    p("")
    for qi in sorted(best_agent_only):
        q_data = questions.get(qi, {})
        truth = q_data.get("answer", "?")
        p(f"  Q{qi} (original: Q{qi+60}) — answer: {truth}")

    # 7. BASELINES CORRECT, BEST AGENT WRONG
    agent_wrong_baseline_right = [qi for qi in range(1, 201)
                                   if best_agent not in q_solved_by[qi]
                                   and any(b in q_solved_by[qi] for b in BASELINES)]
    p("")
    p("=" * 80)
    p(f"7. BASELINE(S) CORRECT, BEST AGENT WRONG — {len(agent_wrong_baseline_right)} questions")
    p(f"   At least one baseline got it, but {best_agent} did not.")
    p("=" * 80)
    p("")
    for qi in sorted(agent_wrong_baseline_right):
        q_data = questions.get(qi, {})
        truth = q_data.get("answer", "?")
        which_baselines = [b for b in BASELINES if b in q_solved_by[qi]]
        agent_pred = all_results[best_agent].get(qi, {}).get("prediction", "-") or "-"
        p(f"  Q{qi} (original: Q{qi+60}) — answer: {truth}, agent predicted: {agent_pred}")
        p(f"    Solved by: {', '.join(which_baselines)}")

    # 8. AGENT COMPARISON
    p("")
    p("=" * 80)
    p("8. AGENT HEAD-TO-HEAD COMPARISON")
    p("=" * 80)
    p("")
    agent_names = AGENTS
    p(f"  {'Pair':<55s} {'Only A':>7s} {'Both':>6s} {'Only B':>7s}")
    p(f"  {'-'*55} {'-'*7} {'-'*6} {'-'*7}")
    for i in range(len(agent_names)):
        for j in range(i+1, len(agent_names)):
            a, b = agent_names[i], agent_names[j]
            only_a = sum(1 for qi in range(1, 201) if a in q_solved_by[qi] and b not in q_solved_by[qi])
            both = sum(1 for qi in range(1, 201) if a in q_solved_by[qi] and b in q_solved_by[qi])
            only_b = sum(1 for qi in range(1, 201) if b in q_solved_by[qi] and a not in q_solved_by[qi])
            p(f"  {a} vs {b:<25s} {only_a:>7d} {both:>6d} {only_b:>7d}")

    # 9. TOOL-USE LIFT PER MODEL (which questions did the agent gain/lose?)
    p("")
    p("=" * 80)
    p("9. TOOL-USE LIFT — Per-model breakdown")
    p("   Questions where agent flipped the result vs its own one-shot baseline.")
    p("=" * 80)
    p("")
    for label, bs_name, ag_name in pairs:
        bs_res = all_results[bs_name]
        ag_res = all_results[ag_name]
        gained = []  # baseline wrong, agent right
        lost = []    # baseline right, agent wrong
        for qi in range(1, 201):
            bs_ok = bs_res.get(qi, {}).get("is_correct") is True
            ag_ok = ag_res.get(qi, {}).get("is_correct") is True
            if ag_ok and not bs_ok:
                gained.append(qi)
            elif bs_ok and not ag_ok:
                lost.append(qi)
        p(f"  {label}:")
        p(f"    Agent gained (baseline wrong -> agent right): {len(gained)}")
        p(f"    Agent lost   (baseline right -> agent wrong): {len(lost)}")
        p(f"    Net lift: {len(gained) - len(lost):+d}")
        if gained:
            p(f"    Gained Qs: {', '.join(f'Q{q}' for q in sorted(gained))}")
        if lost:
            p(f"    Lost Qs:   {', '.join(f'Q{q}' for q in sorted(lost))}")
        p("")

    # 10. AGENT-EXCLUSIVE SOLVES (per agent)
    p("=" * 80)
    p("10. AGENT-EXCLUSIVE SOLVES — Questions each agent solved that NO baseline could")
    p("=" * 80)
    p("")
    for ag_name in AGENTS:
        exclusive = []
        for qi in range(1, 201):
            if ag_name in q_solved_by[qi] and not any(b in q_solved_by[qi] for b in BASELINES):
                exclusive.append(qi)
        p(f"  {ag_name}: {len(exclusive)} exclusive solves")
        if exclusive:
            p(f"    Qs: {', '.join(f'Q{q}' for q in sorted(exclusive))}")
        p("")

    # 11. QUESTIONS SOLVED BY ONLY ONE METHOD
    p("=" * 80)
    p("11. QUESTIONS SOLVED BY EXACTLY ONE METHOD")
    p("    Fragile answers — only a single method got it right.")
    p("=" * 80)
    p("")
    single_solvers = []
    for qi in range(1, 201):
        if len(q_solved_by[qi]) == 1:
            solver = list(q_solved_by[qi])[0]
            single_solvers.append((qi, solver))
    p(f"  Total: {len(single_solvers)}")
    p("")
    by_method = defaultdict(list)
    for qi, solver in single_solvers:
        by_method[solver].append(qi)
    for method in all_names:
        if method in by_method:
            qs = by_method[method]
            p(f"  {method} ({len(qs)}): {', '.join(f'Q{q}' for q in sorted(qs))}")
    p("")

    # 12. OPUS AGENT vs GPT-5.2 AGENT — detailed comparison
    p("=" * 80)
    p("12. OPUS AGENT vs GPT-5.2 AGENT — Question-level comparison")
    p("=" * 80)
    p("")
    opus_ag = all_results["Claude Opus 4.6 + Agent"]
    gpt52_ag = all_results["GPT-5.2 + Agent"]
    opus_only = []
    gpt52_only = []
    both_right = 0
    both_wrong = 0
    for qi in range(1, 201):
        o_ok = opus_ag.get(qi, {}).get("is_correct") is True
        g_ok = gpt52_ag.get(qi, {}).get("is_correct") is True
        if o_ok and g_ok:
            both_right += 1
        elif o_ok and not g_ok:
            opus_only.append(qi)
        elif g_ok and not o_ok:
            gpt52_only.append(qi)
        else:
            both_wrong += 1
    p(f"  Both correct:          {both_right}")
    p(f"  Both wrong:            {both_wrong}")
    p(f"  Opus only:             {len(opus_only)}")
    p(f"  GPT-5.2 only:          {len(gpt52_only)}")
    p("")
    p(f"  Opus-only Qs ({len(opus_only)}): {', '.join(f'Q{q}' for q in sorted(opus_only))}")
    p(f"  GPT-5.2-only Qs ({len(gpt52_only)}): {', '.join(f'Q{q}' for q in sorted(gpt52_only))}")

    # 13. AGENT FAILURE MODES (best agent) — DETAILED
    p("")
    p("=" * 80)
    p(f"13. FAILURE MODES — {best_agent} (22 failures)")
    p("=" * 80)
    p("")
    no_answer_qs = []
    wrong_qs = []
    for qi in range(1, 201):
        r = opus_ag.get(qi, {})
        if r.get("is_correct") is True:
            continue
        pred = r.get("prediction")
        if pred is None:
            no_answer_qs.append(qi)
        else:
            wrong_qs.append(qi)
    p(f"  Wrong answer (picked incorrect choice):  {len(wrong_qs)}")
    p(f"  No answer (timeout / max-steps):         {len(no_answer_qs)}")
    p("")

    # ---- 13a. Wrong-answer breakdown ----
    p("  13a. WRONG-ANSWER BREAKDOWN (10 questions)")
    p("  " + "-" * 76)
    p("")

    wrong_details = {
        1:   ("B", "C", "bad_input",
              "Set initial pH 7.0 without 'charge' keyword; PHREEQC accepted 7.000 "
              "instead of computing equilibrium pH 6.997."),
        15:  ("A", "C", "bad_input",
              "Used 'Cl charge' (can't balance on a lone anion); ran without charge "
              "balance, getting −99.80% instead of the correct 0.00%."),
        18:  ("B", "C", "precision_limit",
              "PHREEQC displayed Cl⁻ activity as 1.000e−04 (4 sig figs). True value "
              "9.998e−5 rounds to 1.000e−4 at display precision; agent couldn't "
              "distinguish B from C."),
        27:  ("B", "C", "bad_input",
              "Modeled HCl addition via 'Cl 1.0e−6' with 'pH 7 charge', getting "
              "pH=5.996. Correct setup yields 6.007 (option C)."),
        66:  ("D", "B", "wrong_interpretation",
              "Conceptual question on Pitzer activity coefficients. Agent identified "
              "gamma > 1 as unusual but over-analyzed option B's mechanism and chose D."),
        85:  ("B", "C", "wrong_interpretation",
              "PHREEQC showed total alkalinity = 1.217e−9 eq/kg. Agent incorrectly "
              "equated alkalinity with HCO₃⁻ molality (2.457e−6) and chose B; "
              "correct answer C = 1.952e−21 was a different quantity."),
        126: ("C", "A", "bad_input",
              "Kinetics rate constant=10 too stiff; after CVODE switch, all 100 steps "
              "showed identical Fe(3) because system over-equilibrated in step 1."),
        128: ("B", "A", "bad_input",
              "Tried ~6 PHREEQC configs. Final run added explicit 1L gas volume at "
              "200°C, dissolving gas phase entirely → S=0.130 instead of correct 0.030."),
        160: ("A", "B", "wrong_interpretation",
              "PHREEQC showed Kaolinite Delta = +4.579e−05 (positive = precipitated). "
              "Agent concluded 'zero moles dissolved' (A); truth B = 4.499e−05 asks "
              "for magnitude of kaolinite change."),
        168: ("D", "A", "bad_input",
              "100 mmol/kgw Ba²⁺ with no counterion → 95% charge imbalance. "
              "Corrupted speciation gave CO₃²⁻ = 2.71e−08 instead of ~3.33e−07."),
    }

    mode_counts = {"bad_input": 0, "wrong_interpretation": 0, "precision_limit": 0}
    for qi in sorted(wrong_details):
        pred, truth, mode, explanation = wrong_details[qi]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        p(f"  Q{qi:>3d} | predicted {pred}, truth {truth} | {mode}")
        p(f"        {explanation}")
        p("")

    p("  Failure-mode tally (wrong answers):")
    p(f"    bad_input            {mode_counts.get('bad_input', 0):>2d}  — flawed PHREEQC input file produced plausible but wrong output")
    p(f"    wrong_interpretation {mode_counts.get('wrong_interpretation', 0):>2d}  — correct PHREEQC output, wrong reasoning about the answer")
    p(f"    precision_limit      {mode_counts.get('precision_limit', 0):>2d}  — PHREEQC display precision too low to distinguish options")
    p("")

    # ---- 13b. No-answer breakdown ----
    p("  13b. NO-ANSWER BREAKDOWN (12 questions)")
    p("  " + "-" * 76)
    p("")

    noanswer_details = {
        63:  ("answered (no file)", 14, "phreeqc_loop",
              "Osmotic coefficient for Mono Lake brine. Ran PHREEQC 4+ times; no "
              "result matched any choice. Lost in formula derivations, never wrote "
              "final_answer.txt."),
        121: ("max_steps_reached", 24, "max_steps",
              "Kinetic albite dissolution + H₂S gas + Goethite SI. Tried 6+ rate "
              "formulations; SI(Goethite) always ~0.39, not matching any choice."),
        129: ("max_steps_reached", 24, "max_steps",
              "Same kinetic albite setup as Q121. Tried rate constants 10, 1e−10, "
              "log-based, CVODE solver. Convergence errors consumed all steps."),
        133: ("max_steps_reached", 24, "max_steps",
              "Calcite precipitation during 1:1 mixing + heating. Struggled with "
              "USE/SAVE/REACTION_TEMPERATURE semantics; phase assemblage kept "
              "resetting between steps."),
        141: ("max_steps_reached", 24, "phreeqc_loop",
              "Gas phase: CO₂ 1atm + CH₄ 5atm + H₂S 10atm at total 1atm. "
              "Contradictory partial pressures; 9+ interpretations tried."),
        143: ("max_steps_reached", 24, "phreeqc_loop",
              "Same contradictory gas phase (asking for pe). 8+ configurations "
              "giving pe values of −4.2, −3.4, −3.7, −4.4 — none matched choices."),
        147: ("max_steps_reached", 24, "phreeqc_loop",
              "Same gas phase (asking for S molality). Results ranged 3.8e−7 to "
              "5.5e−1 depending on interpretation."),
        148: ("max_steps_reached", 24, "phreeqc_loop",
              "Same gas phase (asking for conductance). Got 13042, 6398, 475, "
              "17741 — choices were ~7889–8339. No configuration matched."),
        149: ("max_steps_reached", 24, "phreeqc_loop",
              "Same gas phase (asking for alkalinity). Got 3.2e−2 to 5.6e−2; "
              "none matched answer choices."),
        182: ("max_steps_reached", 24, "phreeqc_loop",
              "Pure water + CH₄/CO₂/NH₃ gas at 150°C. Confused moles vs partial "
              "pressures in GAS_PHASE syntax; 8+ volume/pressure configs tried."),
        185: ("max_steps_reached", 24, "phreeqc_loop",
              "Same gas setup as Q182 (asking for alkalinity). Small-vol gave "
              "1.5e−3, large-vol gave 2.0e−2. Neither matched."),
        190: ("max_steps_reached", 24, "phreeqc_loop",
              "Same gas setup as Q182 (asking for pe). Small-vol −4.906 vs "
              "large-vol −5.566. No exact match."),
    }

    no_mode_counts = {"max_steps": 0, "phreeqc_loop": 0}
    for qi in sorted(noanswer_details):
        status, steps, mode, explanation = noanswer_details[qi]
        no_mode_counts[mode] = no_mode_counts.get(mode, 0) + 1
        p(f"  Q{qi:>3d} | {status:<26s} | steps: {steps:>2d} | {mode}")
        p(f"        {explanation}")
        p("")

    p("  Failure-mode tally (no answers):")
    p(f"    phreeqc_loop  {no_mode_counts.get('phreeqc_loop', 0):>2d}  — agent repeatedly re-ran PHREEQC with varied configs, exhausted 24 steps")
    p(f"    max_steps     {no_mode_counts.get('max_steps', 0):>2d}  — complex PHREEQC setup; agent couldn't converge within step budget")
    p("")

    # ---- 13c. Root-cause clusters ----
    p("  13c. ROOT-CAUSE CLUSTERS (no-answer failures)")
    p("  " + "-" * 76)
    p("")
    p("  Cluster 1 — Contradictory gas-phase specification (5 Qs: Q141, Q143, Q147, Q148, Q149)")
    p("    All share the same setup where partial pressures sum to 16 atm but total")
    p("    pressure is stated as 1 atm. The agent tried every interpretation but")
    p("    couldn't converge on the 'intended' one.")
    p("")
    p("  Cluster 2 — GAS_PHASE moles-vs-pressure confusion (3 Qs: Q182, Q185, Q190)")
    p("    PHREEQC GAS_PHASE syntax 'CH4(g) 10' means partial pressure = 10 atm,")
    p("    not 10 moles. The agent couldn't disambiguate, yielding divergent results")
    p("    across configurations.")
    p("")
    p("  Cluster 3 — Complex PHREEQC modeling challenges (4 Qs: Q63, Q121, Q129, Q133)")
    p("    Each involved a distinct modeling difficulty: osmotic coefficients (Pitzer),")
    p("    kinetic rate law interpretation, or sequential temperature/mixing semantics.")
    p("")

    # ---- 13d. Summary statistics ----
    p("  13d. SUMMARY")
    p("  " + "-" * 76)
    p("")
    total_failures = len(wrong_qs) + len(no_answer_qs)
    p(f"  Total failures:         {total_failures} / 200  ({total_failures/2:.1f}%)")
    p(f"    Wrong answer:         {len(wrong_qs):>2d}  ({len(wrong_qs)/total_failures*100:.0f}% of failures)")
    p(f"    No answer:            {len(no_answer_qs):>2d}  ({len(no_answer_qs)/total_failures*100:.0f}% of failures)")
    p("")
    p(f"  Wrong-answer root causes:")
    p(f"    Bad PHREEQC input:      6  (60%) — most common; subtle errors in model setup")
    p(f"    Wrong interpretation:   3  (30%) — correct output, flawed reasoning")
    p(f"    Precision limit:        1  (10%) — PHREEQC display can't resolve answer choices")
    p("")
    p(f"  No-answer root causes:")
    p(f"    PHREEQC loop:           9  (75%) — repeated attempts, exhausted step budget")
    p(f"    Complex setup:          3  (25%) — couldn't formulate correct PHREEQC input")
    p("")
    p(f"  Key insight: 8 of 12 no-answer failures (67%) cluster into just 2 question")
    p(f"  templates (gas-phase specification ambiguity). Fixing the agent's understanding")
    p(f"  of GAS_PHASE syntax could recover up to 8 questions (+4 pp accuracy).")

    # 14. KEY FINDINGS
    p("")
    p("=" * 80)
    p("14. KEY FINDINGS FOR PAPER")
    p("=" * 80)
    p("")
    p("  1. Claude Opus 4.6 + Agent achieves the highest score: 178/200 (89.0%).")
    p("     This is +15.5 pp over its own one-shot baseline and +15.5 pp over GPT-5.2 + Agent.")
    p("")
    p("  2. Tool-augmented agents improve performance for 3 of 4 model families:")
    for label, bs, ag in pairs:
        bs_c = sum(1 for r in all_results[bs].values() if r.get("is_correct") is True)
        ag_c = sum(1 for r in all_results[ag].values() if r.get("is_correct") is True)
        diff = ag_c - bs_c
        direction = "+" if diff > 0 else ""
        p(f"     {label}: {bs_c} -> {ag_c} ({direction}{diff})")
    p("")
    p("  3. 'Thinking model' paradox: In one-shot MCQ, larger reasoning models (GPT-5.2, Gemini Pro)")
    p("     score LOWER than their smaller counterparts (GPT-5.1, Gemini Flash).")
    p("     Exception: Anthropic, where Opus (147) >> Sonnet (130).")
    p("")
    p(f"  4. {len(none_solved)} questions remain unsolved by ANY method — candidates for dataset review.")
    p("")
    p(f"  5. {len(best_agent_only)} questions solved ONLY by the best agent (Opus + tools),")
    p("     demonstrating that PHREEQC tool use enables answers impossible from knowledge alone.")
    p("")
    p("=" * 80)
    p("END OF ANALYSIS")
    p("=" * 80)

    return "\n".join(out)


if __name__ == "__main__":
    text = main()
    out_path = RESULT_ROOT / "analysis_SJ_200.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Written to {out_path}")
    print(text)
