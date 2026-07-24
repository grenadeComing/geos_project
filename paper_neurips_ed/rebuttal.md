# Response to Reviewer 3hnF

Thank you for recognizing the value of the kept/gained/lost decomposition and capability-stratified failure analysis.

**Scope and relation to prior work.** We agree that the present evidence is a controlled case study of one deterministic simulator, not evidence about arbitrary multi-tool scientific workflows. We will narrow the title and remove the statement that the capability-tier interaction “likely transfers” to other simulators. A second simulator would require new domain-expert item construction and validation and cannot be added responsibly during rebuttal. Relative to ScienceAgentBench and ToRA, our contribution is not the already-known observation that tools can help. The deterministic, paired setting exposes effects hidden by aggregate accuracy: which previously correct items are lost, whether protocol gaps exceed rerun variation, and whether failure occurs during simulator construction, execution, output navigation, or final commitment. We will make this distinction explicit.

**Numeric-answer evaluation.** We added a numeric-answer version of every item with an unambiguous scalar target (182/200), removing answer labels and distractors while preserving the underlying simulator task. Results on this same 182-item set are:

| Answer format | Direct | CoT | Agent-TOC + PHREEQC | Tool vs. better no-tool |
| --- | ---: | ---: | ---: | ---: |
| MCQ | 48/182 (26.4%) | 60/182 (33.0%) | 131/182 (72.0%) | +39.0 pp |
| Numeric/FIB | 2/182 (1.1%) | 4/182 (2.2%) | 93/182 (51.1%) | +48.9 pp |

Removing the choices lowers absolute accuracy, as expected, but does not explain the tool-access result: the advantage over the stronger no-tool baseline increases from 39.0 to 48.9 pp. The numeric tool run grades all 182 items with zero execution errors (95% Wilson CI 43.9–58.3%). Relative to Direct it gains 91 items and loses none; relative to CoT it gains 90 and loses one. We updated the anonymous repository with the dataset, targets/tolerances, per-item predictions, complete workspaces/logs, and a one-command analysis; further implementation and grading details are available there.

**Effective size, reuse, human baseline, and tier labels.** The 200 items come from 21 scenario/setup families; multiple items may query different fields or time steps from one setup. We will state this prominently and retain scenario-clustered uncertainty. Every item is evaluated in a separate isolated agent session, so no simulation or workspace is reused across questions. We cannot provide an independent human baseline: the same domain-qualified co-author authored and simulator-checked the items, so evaluating that person would introduce direct item-construction leakage and would not constitute an independent baseline. This co-author is a doctoral researcher in civil/environmental engineering, with undergraduate training in earth/environmental sciences and mathematics and prior research in subsurface flow, geologic carbon storage, and computational geoscience. We will report the lack of an independent baseline rather than make a near-human claim. Finally, “top/mid/below-mid” will be defined only as empirical groupings induced by the observed protocol and failure patterns in this benchmark, not intrinsic model-family rankings.

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

**1. Are complete per-question runs now available for every condition?** No—not for every legacy condition. Some original condition-specific trajectories were not retained, and we cannot recreate an original trajectory by rerunning a changed closed model. We therefore make the availability boundary explicit. We restored the two accidentally omitted *original* complete Gemini 3 Flash runs (TOC: 47/200; Raw100k: 94/200); their summaries and per-question logs record the original May 3, 2026 execution times, while the later Git commit records only the packaging correction. We also release all other retained original trajectories, including Opus Raw100k, three complete Opus TOC reruns, five 45-item TOC tail runs, external-harness runs, and original failure-case workspaces, plus complete trajectories for every rebuttal-period run. These original materials provide independently inspectable evidence of the released harness, tool calls, simulator execution, log schema, and failure behavior, although they do not substitute for a missing condition-specific trajectory.

For the legacy conditions whose trajectories were not retained, we instead release the exact per-item correctness vectors or aggregate sufficient statistics preserved from the original analysis. These records make the published numerical comparisons reproducible but are explicitly labeled as derived records, not restored logs. We will correct the manuscript's overly broad trajectory-availability claim and distinguish **numerical reproducibility** from **trajectory availability** throughout.

**2. What exact commands now work?** We fixed the dataset and `result`/`results` path mismatches and removed the analysis guard. From the anonymous repository root,

```bash
node analysis/analyze_results.js
```

now analyzes every complete run physically present in the release and records its path and integrity status. To regenerate the previously non-reproducible published tables, run:

```bash
node analysis/reproduce_published_tables.js
```

This regenerates every numerical cell in Tables 2, 3, and 5 and reports **113/113 checks passed**. Table 2 is recomputed from complete Direct outcomes and a lossless 1,000-row Agent-TOC/Raw100k correctness snapshot; Tables 3 and 5 use the retained correct-count and exact input-token sufficient statistics. Input provenance, checksums, and the artifact-update history are documented in the repository.

**3. Title and scope.** We agree that the original title generalizes beyond the evidence. We will revise it to *“PHREEQC-MCQ-200: A Diagnostic Benchmark for Tool-Augmented LLM Agents on a Geochemical Simulator”* and frame the empirical conclusions as a PHREEQC case study rather than evidence about all scientific simulators.

**4. How was expert review performed?** The original paper and Croissant metadata overstated the procedure. One domain-qualified co-author authored and simulator-checked the items. This co-author is a doctoral researcher in civil/environmental engineering, with undergraduate training in earth/environmental sciences and mathematics and prior research in subsurface flow, geologic carbon storage, and computational geoscience. There was no independent expert panel, agreement statistic, or recorded rejection rate. We corrected the metadata and will replace “expert-validated” with “expert-authored and simulator-verified.” We will state the executable acceptance criteria: successful execution with the frozen simulator/database, a unique target supported by the output, plausible distractors, and question clarity. These checks establish simulator-consistent ground truth, not independent expert validation; the latter remains future work.

**5. Do labels survive PHREEQC or database revisions?** We did not test this and do not claim version invariance. Ground truth is defined for PHREEQC 3.8.6 with the released `phreeqc.dat`. We will add explicit version/database provenance and state that label stability under another PHREEQC point release or thermodynamic database revision is unknown.

**6. Why no open-weight models, and does the tier interaction extend to them?** The evaluation budget was allocated to paired protocol comparisons across six frontier models from three vendors, so open-weight models were omitted. Consequently, we have no evidence that the observed capability-by-interface interaction extends to open-weight models. We will scope the claim to the evaluated systems and identify open-weight evaluation as future work.
