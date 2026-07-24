# Response to Reviewer 3hnF

Thank you for recognizing the value of the kept/gained/lost decomposition and capability-stratified failure analysis.

**Scope and relation to prior work.** We agree that the present evidence is a controlled case study of one deterministic simulator, not evidence about arbitrary multi-tool scientific workflows. We will narrow the title and remove the statement that the capability-tier interaction “likely transfers” to other simulators. A second simulator would require new domain-expert item construction and validation and cannot be added responsibly during rebuttal. Relative to ScienceAgentBench and ToRA, our contribution is not the already-known observation that tools can help. The deterministic, paired setting exposes effects hidden by aggregate accuracy: which previously correct items are lost, whether protocol gaps exceed rerun variation, and whether failure occurs during simulator construction, execution, output navigation, or final commitment. We will make this distinction explicit.

**Numeric-answer evaluation.** We added a numeric-answer version of every item with an unambiguous scalar target (182/200), removing answer labels and distractors while preserving the underlying simulator task. Results on this same 182-item set are:

| Answer format | Direct | CoT | Agent-TOC + PHREEQC | Tool vs. better no-tool |
| --- | ---: | ---: | ---: | ---: |
| MCQ | 48/182 (26.4%) | 60/182 (33.0%) | 131/182 (72.0%) | +39.0 pp |
| Numeric/FIB | 2/182 (1.1%) | 4/182 (2.2%) | 93/182 (51.1%) | +48.9 pp |

Removing the choices lowers absolute accuracy, as expected, but does not explain the tool-access result: the advantage over the stronger no-tool baseline increases from 39.0 to 48.9 pp. The numeric tool run grades all 182 items with zero execution errors (95% Wilson CI 43.9–58.3%). Relative to Direct it gains 91 items and loses none; relative to CoT it gains 90 and loses one. We updated the anonymous repository with the dataset, targets/tolerances, per-item predictions, complete workspaces/logs, and a one-command analysis; further implementation and grading details are available there.

**Effective size, reuse, human baseline, and tier labels.** The 200 items come from 21 scenario/setup families; multiple items may query different fields or time steps from one setup. We will state this prominently and retain scenario-clustered uncertainty. Every item is evaluated in a separate isolated agent session, so no simulation or workspace is reused across questions. We cannot provide an independent human baseline: the only available aqueous-geochemistry expert authored and simulator-checked the items, so evaluating that same person would introduce direct item-construction leakage and would not constitute an independent baseline. We will report this limitation rather than make a near-human claim. Finally, “top/mid/below-mid” will be defined only as empirical groupings induced by the observed protocol and failure patterns in this benchmark, not intrinsic model-family rankings.

# Response to Reviewer MmtE

Thank you for highlighting the diagnostic value of the benchmark and for suggesting the two most useful robustness checks.

**Numeric/free-response subset.** We added a numeric-answer version of every item with an unambiguous scalar target (182/200). On the same items:

| Answer format | Direct | CoT | Agent-TOC + PHREEQC | Tool vs. better no-tool |
| --- | ---: | ---: | ---: | ---: |
| MCQ | 48/182 (26.4%) | 60/182 (33.0%) | 131/182 (72.0%) | +39.0 pp |
| Numeric/FIB | 2/182 (1.1%) | 4/182 (2.2%) | 93/182 (51.1%) | +48.9 pp |

Thus removing choices makes the task harder but strengthens the relative tool/no-tool separation. This directly tests recovery of the scientific value before A/B/C/D mapping and rules out MCQ shortcuts as the explanation for the main tool-access finding. We updated the anonymous repository with all prompts, targets/tolerances, predictions, complete agent workspaces/logs, and grading code; further details are available there.

**Harness and 24-step ceiling.** We agree that the result measures a compound system, including the harness and interface. We will make that scope explicit. The existing all-model trajectory evidence also separates the relevant mechanisms: 0% of GPT-5.1 and 9.7% of GPT-5.2 failures reach the 24-step ceiling, compared with 62–68% of top-tier failures; the mid-tier failed-item median step counts (7–9) are approximately their overall medians. Moreover, the prompt does not disclose the step budget. These observations argue that the mid-tier TOC regression is primarily premature commitment/navigation behavior rather than ceiling exhaustion. For Gemini 3 Flash, in contrast, we explicitly attribute the TOC regression to interface overhead and non-commitment (median nine TOC reads versus two reads under Raw100k). We will not generalize the fixed-budget result beyond this harness, and leave broader budget/retrieval-granularity ablations to future work.

**Repeated Raw100k runs.** We completed the requested independent Raw100k repeats. GPT-5.1 changes from 127/200 (63.5%) to 122/200 (61.0%; 95% Wilson CI 54.1–67.5%), and GPT-5.2 from 123/200 (61.5%) to 127/200 (63.5%; 56.6–69.9%). Both repeats remain above the entire corresponding TOC rerun ranges (GPT-5.1: 48.0–55.5%; GPT-5.2: 54.5–56.0%). Aggregate changes are only -2.5 and +2.0 pp, although 59–60 item outcomes flip: GPT-5.1 has 95 both-correct / 27 repeat-only / 32 original-only / 46 both-wrong outcomes, and GPT-5.2 has 95 / 32 / 28 / 45. This reinforces rather than weakens our distinction between aggregate stability and item-level retention. Both complete 200-item repeat runs and the paired analysis are released. Claims about Raw500k and external harnesses will remain explicitly single-run findings.

# Response to Reviewer h7Cb

Thank you for running the artifact and identifying concrete release failures. The reviewer is correct that the submitted package was incomplete; we do not claim that the omitted files were available at review time.

**Artifact correction.** We fixed the dataset and `result`/`results` path mismatches and removed the analysis guard. We also added a dedicated published-table reproduction command; from the anonymous repository root, run:

```bash
node analysis/reproduce_published_tables.js
```

This regenerates every numerical cell in Tables 2, 3, and 5 and reports 113/113 checks passed. Table 2 is recomputed from complete Direct outcomes and a lossless 1,000-row Agent-TOC/Raw100k correctness snapshot; Tables 3 and 5 are recomputed from the retained correct-count and input-token sufficient statistics. We also restored both original complete 200-item Gemini 3 Flash runs: Agent-TOC is 47/200 (23.5%) and Raw100k is 94/200 (47.0%). Their internal metadata/logs record the original May 3, 2026 executions; the later Git commit records only the packaging correction. Importantly, the missing trajectories are condition-specific rather than artifact-wide: the release retains complete per-question workspaces/logs for original custom-agent evidence including Opus Raw100k, three Opus TOC reruns, five 45-item TOC runs, and both Gemini protocols, as well as the external-harness evaluations, extensive original failure-case workspaces, and every rebuttal-period run. These retained runs directly expose the harness, tool calls, logging schema, and observed failure behavior. Checksums, input provenance, and an artifact-update manifest are included. For the specified legacy conditions whose complete workspaces were not retained, we distinguish numerical reproducibility from trajectory availability: the numerical tables now reproduce exactly, but derived correctness/sufficient-statistic snapshots are not presented as restored trajectories. We will correct the manuscript’s overly broad trajectory-availability claim accordingly.

**Expert validation.** The original wording and Croissant metadata overstated the procedure. One aqueous-geochemistry expert authored and simulator-checked the items; there was no independent expert panel, inter-rater agreement statistic, rejection rate, or independent human baseline. We have corrected the metadata and will replace “expert-validated” with “expert-authored and simulator-verified.” We will document the executable acceptance criteria: successful execution with the frozen simulator/database, a unique target supported by the output, plausible distractors, and question clarity. Independent expert validation remains future work.

**Scope, version stability, and open-weight models.** We will narrow the framing to a PHREEQC case study. Ground truth is defined for the pinned PHREEQC 3.8.6 binary and bundled `phreeqc.dat`; label invariance under other point releases or database revisions was not tested and will be stated as a limitation. Open-weight models were excluded by the finite evaluation budget, so the observed capability grouping cannot be claimed to cover them.
