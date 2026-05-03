# ACM CAIS 2026 Artifact Snapshot

This directory is a **frozen snapshot** of the artifacts for our ACM CAIS 2026
submission (Paper #322, *"PHREEQC-Agent: Tool-Augmented LLM for Geochemical
Computation"*), captured at the time of the author-response submission.

Nothing in here should be modified. The active code for the follow-up project
lives at the repository root alongside a fresh `result/` and `datasets/`.

## Layout

```
archive_acm_2026/
├── README.md                                   # this file
├── dataset_Sachit+Jerry_for_ACM_paper.jsonl    # 200-question MCQ benchmark (S + J)
├── experiment_results.md                       # full rebuttal draft + raw data
├── analysis_SJ_200.txt                         # per-option / failure analysis
├── paper.txt                                   # plaintext of the submitted paper
├── sj_files.zip                                # archived dataset bundle
├── database_reference.txt                      # PHREEQC database reference (recovered from git)
├── phreeqc.dat                                 # PHREEQC thermodynamic database (recovered from git)
├── figs/                                       # figures used in the paper
├── scripts/                                    # ACM-specific scripts
│   ├── baseline_anthropic.py                   # one-shot Anthropic baseline runner (recovered from git)
│   ├── baseline_google.py                      # one-shot Google baseline runner (recovered from git)
│   ├── baseline_gpt.py                         # one-shot OpenAI baseline runner (recovered from git)
│   ├── evaluate.py                             # shared evaluation utilities (recovered from git)
│   ├── analyze_all.py                          # aggregate analysis script (recovered from git)
│   ├── evaluate_ablation_raw.py                # TOC vs Raw ablation (Exp 2)
│   ├── evaluate_claude_sdk.py                  # SDK comparison runner
│   ├── aggregate_sdk_comparison.py             # SDK comparison aggregator
│   ├── evaluation_chart.jsx                    # chart component for paper figure
│   └── collect_review.py                       # early review-collection helper
└── result/                                     # all per-run outputs
    ├── oneshot/                                # one-shot baselines (paper Table 1)
    │   ├── sj_baseline_claude_opus46/          # Opus 4.6
    │   ├── sj_baseline_claude_sonnet46/        # Sonnet 4.6
    │   ├── sj_baseline_gpt51/                  # GPT-5.1
    │   ├── sj_baseline_gpt52/                  # GPT-5.2
    │   ├── sj_baseline_gpt54/                  # (exploratory)
    │   ├── sj_baseline_gemini25flash/          # Gemini 2.5 Flash
    │   └── sj_baseline_gemini25pro/            # Gemini 2.5 Pro
    ├── cot/                                    # CoT baselines (rebuttal Exp 1)
    │   ├── sj_cot_claude_opus46/
    │   ├── sj_cot_claude_sonnet46/
    │   ├── sj_cot_gpt51/
    │   └── sj_cot_gpt52/
    ├── agent/                                  # agent runs (paper Table 1 + Exp 2)
    │   ├── sj_agent_claude_opus46/             # original paper run (no token logs)
    │   ├── sj_agent_gpt51/                     # original paper run
    │   ├── sj_agent_gpt52/                     # original paper run
    │   ├── sj_agent_gemini25flash/             # original paper run
    │   ├── sj_toc_opus46/                      # rebuttal re-run (TOC, with tokens)
    │   ├── sj_toc_sonnet46/                    # rebuttal Sonnet TOC
    │   ├── sj_toc_gpt51/                       # rebuttal re-run
    │   ├── sj_toc_gpt52/                       # rebuttal re-run
    │   ├── sj_ablation_raw_opus46/             # rebuttal Raw ablation
    │   ├── sj_raw_sonnet46/                    # rebuttal Sonnet Raw
    │   ├── sj_ablation_raw_gpt51/              # rebuttal Raw ablation
    │   └── sj_ablation_raw_gpt52/              # rebuttal Raw ablation
    └── claude_sdk/                             # Claude Agent SDK comparison
        ├── sdk_opus46_v2/                      # Opus full 200Q run
        ├── sdk_sonnet46_v1/                    # Sonnet full 200Q run
        └── mini_*/                             # 2-question mini runs (with tokens)
```

## Key result mappings

| Paper / rebuttal source | Run directory |
|---|---|
| Paper Table 1 — one-shot | `result/oneshot/sj_baseline_*/summary.json` |
| Paper Table 1 — agent    | `result/agent/sj_agent_*/summary.json` (original, no token logs) |
| Rebuttal Exp. 1 (CoT)    | `result/cot/sj_cot_*/summary.json` |
| Rebuttal Exp. 2 (TOC re-run) | `result/agent/sj_toc_*/summary.json` |
| Rebuttal Exp. 2 (Raw ablation) | `result/agent/sj_ablation_raw_*/`, `sj_raw_sonnet46/summary.json` |
| Rebuttal (Claude Agent SDK)    | `result/claude_sdk/sdk_*/summary.json` |

## Why this is frozen

The follow-up project (TOC as a general context-management pattern, targeting a
top-venue submission) uses:

- A **revised benchmark** built from Sachit's updated 100 questions (Jerry's
  100 simpler questions are dropped).
- A **new baseline set** (Full / TOC / Summary / RAG) including the baselines
  a reviewer would expect at a top venue.
- An **MCQ + Fill-in-the-blank** dual task format.

Mixing the old and new results would make comparability impossible, so the
ACM snapshot is preserved intact and the new work starts from a clean
`result/` and `datasets/` at the repo root.
