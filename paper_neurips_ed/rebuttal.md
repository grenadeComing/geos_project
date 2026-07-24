# Response to Reviewer 3hnF

Thank you for recognizing the value of the kept/gained/lost decomposition and capability-stratified failure analysis.

**1. Effective benchmark size: how many distinct PHREEQC runs underlie the 200 items, and are runs reused?** There are **20 distinct underlying simulation configurations**. Each simulation supports 8--19 questions that read different output fields, reaction steps, or derived quantities from the same run. The manuscript's count of 21 refers to exact textual stems: one sulfuric-acid setup was split into two stems because one item adds the words “to solution 1,” although both specify the same PHREEQC input. We will correct this distinction and merge those two groups in the clustered analysis. The evaluation does not exploit this shared structure: every question starts a fresh isolated agent session and workspace, so the logs show agents independently reconstructing the setup for each item rather than reusing one execution across questions. Retries within an item's session are retained in the logs.

**2. Human expert calibration: what accuracy do experts achieve?** We do not have an independent human baseline and will not use the model scores to imply near-human performance. The one domain-qualified co-author who authored and simulator-checked the items cannot provide an unbiased baseline because that would introduce direct item-construction leakage. This author is a doctoral researcher in civil/environmental engineering, with undergraduate training in earth/environmental sciences and mathematics and prior computational-geoscience research.

**3. Generalization: has the interaction been tested on another simulator?** No. We will remove the statement that the interaction “likely transfers” and scope the empirical claim to PHREEQC. A second simulator benchmark requires a new domain expert and independently constructed, simulator-verified items; substituting an unvalidated second domain would weaken rather than strengthen the evaluation. We did add a generic keyword-search interface that uses no PHREEQC section headers. The same within-PHREEQC pattern persists: Sonnet scores 84.0/83.0/83.5% under TOC/Grep/Raw100k, while GPT-5.2 scores 54.0/51.0/61.5% and GPT-5.1 scores 54.0/46.5/63.5%. This shows that the result is not specific to the TOC regex, while leaving cross-simulator generalization open.

**MCQ-format concern.** We converted every item with an unambiguous scalar target (182/200) to numeric free response and evaluated GPT-5.4 under Direct, CoT, and Agent-TOC, removing labels and distractors while preserving the simulator task:

| Answer format | Direct | CoT | Agent-TOC + PHREEQC | Tool vs. better no-tool |
| --- | ---: | ---: | ---: | ---: |
| MCQ | 48/182 (26.4%) | 60/182 (33.0%) | 131/182 (72.0%) | +39.0 pp |
| Numeric/FIB | 2/182 (1.1%) | 4/182 (2.2%) | 93/182 (51.1%) | +48.9 pp |

Removing choices lowers absolute accuracy but increases the advantage over the stronger no-tool baseline from +39.0 to +48.9 pp. Relative to Direct, Agent-TOC gains 91 items and loses none; relative to CoT, it gains 90 and loses one. Thus distractor elimination or answer-position preference cannot explain the main tool-access finding. The complete dataset, targets/tolerances, predictions, workspaces/logs, and grading code are released.

**Controlled-setting contribution and tier labels.** Our contribution is not the already-known observation that tools can help. Holding the simulator, database, questions, and grading fixed enables direct paired measurement of lost no-tool-correct items, protocol gaps relative to rerun variation, and failure stage. Multi-tool selection and composition are complementary questions; this benchmark deliberately isolates reliability after the simulator has been selected. The tier labels are descriptive shorthand for the models' generation and market positioning at the time of evaluation, not groups assigned after observing PHREEQC scores: “top” comprises the then-current frontier models (Claude Opus 4.6, Claude Sonnet 4.6, and GPT-5.4), “mid” comprises the preceding GPT-5.1/5.2 generation, and “below-mid” denotes the lighter-weight Gemini 3 Flash Preview. We will add this definition and the exact group membership to Table 1's caption.

# Response to Reviewer MmtE

Thank you for highlighting the benchmark's diagnostic value. We completed the requested numeric-answer evaluation and Raw100k repeats, and added a model-agnostic retrieval ablation.

**1. Does the result survive numeric/free-response evaluation? Yes.** We converted every item with an unambiguous scalar target (182/200), rather than only a small subset, and evaluated GPT-5.4 under Direct, CoT, and Agent-TOC. On these same items:

| Answer format | Direct | CoT | Agent-TOC + PHREEQC | Tool vs. better no-tool |
| --- | ---: | ---: | ---: | ---: |
| MCQ | 48/182 (26.4%) | 60/182 (33.0%) | 131/182 (72.0%) | +39.0 pp |
| Numeric/FIB | 2/182 (1.1%) | 4/182 (2.2%) | 93/182 (51.1%) | +48.9 pp |

Removing choices makes the task harder, but the tool advantage over the stronger no-tool baseline increases from +39.0 to +48.9 pp. Relative to Direct, Agent-TOC gains 91 items and loses none; relative to CoT, it gains 90 and loses one. Thus option selection cannot explain the main tool-access result. The dataset, targets/tolerances, predictions, complete workspaces/logs, and grading code are released.

**2. Is the TOC capability-tier effect only a consequence of the 24-step budget or PHREEQC-specific section retrieval? No for the mid-tier comparison; Gemini remains budget-specific.** We added a generic Agent-Grep protocol that uses keyword search plus ranged reads, with the same agent loop, 24-step budget, and grading:

| Model | Agent-TOC | Agent-Grep | Agent-Raw100k |
| --- | ---: | ---: | ---: |
| Claude Sonnet 4.6 | 84.0% | 83.0% | 83.5% |
| GPT-5.2 | 54.0% | 51.0% | 61.5% |
| GPT-5.1 | 54.0% | 46.5% | 63.5% |

Sonnet remains within 1 pp across both retrieval interfaces and Raw100k, whereas both mid-tier GPT models remain well below Raw100k under either retrieval interface. The same tier pattern therefore persists without PHREEQC's section-level TOC design. Nor is the mid-tier gap caused by the step ceiling: only 0% of GPT-5.1 and 9.7% of GPT-5.2 failed TOC items reach step 24; their dominant failure is early wrong commitment. Gemini 3 Flash is different: 117/200 TOC trajectories reach the ceiling versus 57/200 under Raw100k. We therefore state its degradation specifically as interface overhead under this 24-step harness, not as an intrinsic model property.

**3. Do the key TOC-versus-Raw100k comparisons survive new independent repeats? Yes.** On July 23, 2026, we reran the two models requested and released both complete 200-item runs:

| Model | Original Raw100k | Raw100k repeat | TOC rerun range |
| --- | ---: | ---: | ---: |
| GPT-5.1 | 63.5% | 61.0% | 48.0–55.5% |
| GPT-5.2 | 61.5% | 63.5% | 54.5–56.0% |

Both Raw100k repeats remain above the entire corresponding TOC rerun range; each original Raw100k score also lies within the repeat's 95% Wilson CI. At the same time, 59–60 per-item outcomes change, reinforcing the paper's distinction between aggregate stability and item-level reliability. Both complete 200-item repeat runs and the paired analysis are released. We will label Raw500k and external-harness conclusions as single-run results.

# Response to Reviewer h7Cb

Thank you for running the artifact and identifying concrete release failures. The reviewer is correct that the submitted package was incomplete; we do not claim that the omitted files were available at review time.

**1. Are complete per-question runs now available for every condition?** Most original experimental evidence was retained. There are three simple cases: (1) the files were included in the originally submitted anonymous artifact; (2) we had the original files but forgot to include them in that package, and have now added them; or (3) the complete logs were not retained, but the pre-submission failure analysis preserved the exact per-item results.

Fortunately, for the last category, we retained the corresponding pre-submission failure-analysis packages. Each failure directory identifies one incorrect item; those item IDs and their complement preserve the exact per-item correctness vector. Thus, although the complete prompt/tool-call logs were lost for a small number of legacy conditions, the experimental numbers were not lost. These are retained original analysis records—not estimates inferred backward from the aggregate values printed in the paper. The released verification script confirms exact item-ID agreement between the failure analysis and saved outcome snapshots for 12 model/protocol conditions.

| Result | Status |
| --- | --- |
| **Table 1** | The Direct/CoT and Opus Raw100k files were in the original anonymous artifact. For the remaining Agent conditions, the complete logs were not retained, but their exact per-item results were preserved in the failure analysis. |
| **Table 2** | The Direct files were in the original anonymous artifact. The Agent per-item results were preserved in the failure analysis, which exactly determines every kept/gained/lost count. |
| **Table 3** | We forgot to package 12 of the 15 original TOC reruns. We have now added them, so all 15 original reruns are released. The embedded timestamps in their run summaries and per-item logs date the runs to May 3–7, 2026, before submission, rather than the rebuttal period. |
| **Table 4** | All five 45-item runs were in the original anonymous artifact. Their baseline per-item results were preserved in the failure analysis. |
| **Table 5** | The complete logs were not retained, but the exact results were preserved in the failure analysis together with the original bin/token statistics. |
| **Gemini §4.3** | We forgot to package both complete original Gemini runs. We have now added them. The embedded timestamps in their run summaries and per-item logs date both runs to May 3, 2026, before submission, rather than the rebuttal period. |

We will revise the manuscript to distinguish complete logs from retained per-item outcomes. Rerunning a missing condition is technically straightforward, but closed-model drift means that it would provide a new complete log set rather than reproduce the exact original trajectories or score. Our two rebuttal-period Raw100k repeats illustrate this: aggregate accuracy changed by only −2.5 pp (GPT-5.1) and +2.0 pp (GPT-5.2), with each original score inside the corresponding repeat's 95% Wilson CI, even though 59–60 per-item outcomes changed. The artifact already contains many complete trajectory sets across the same harness and protocols. A full replacement sweep is beyond our budget (one Opus Raw100k run alone used 46.77M input and 0.37M output tokens); however, if the reviewer considers one specific missing condition essential, please identify it and we will rerun that condition and release its complete logs.

**2. What is the exact reproduction command?** From the anonymous repository root, run:

```bash
node analysis/reproduce_published_tables.js
```

It regenerates every previously non-reproducible numerical cell in Tables 2, 3, and 5 and reports **113/113 checks passed**. We also fixed the path errors and removed the guard, so `node analysis/analyze_results.js` now runs directly on the released layout.

**3. How many experts authored and reviewed the items?** One domain-qualified co-author authored and simulator-checked all items. This author is a doctoral researcher in civil/environmental engineering, with undergraduate training in earth/environmental sciences and mathematics and prior computational-geoscience research. There was no independent second expert; therefore, there is no inter-expert agreement statistic or recorded rejection rate. We will replace “expert-validated” with “expert-authored and simulator-verified.”

**4. Do labels survive PHREEQC or database revisions?** We have not run a cross-version label check, so we do not claim invariance. The released database was not created specifically for this benchmark: its header identifies it as the PHREEQC `Amm.dat` snapshot dated May 22, 2024, released under the filename `phreeqc.dat` and frozen for evaluation. `Amm.dat` closely follows the standard `phreeqc.dat`, principally decoupling ammonia redox; the relevant 2024 upstream revisions also concern viscosity, specific conductance, and diffusion. Sensitivity to these differences is therefore more likely to be concentrated in ammonia/redox and related transport observables than across the entire benchmark. Nevertheless, small numerical changes could flip close answer choices. Ground truth is therefore defined for PHREEQC 3.8.6 with the released database. All reported model comparisons use this same fixed target, while cross-version/database label stability remains untested and will be stated as a limitation.

**5. Why no open-weight models, and does the result extend to them?** Open-weight models were outside our original evaluation scope, not excluded by the method. We spent the available inference budget on full paired evaluations of six then-frontier models from three vendors, including tool/no-tool, TOC/Raw, and repeated-run comparisons. The harness is model-agnostic: evaluating an open-weight model requires only a new model-call adapter; the benchmark, PHREEQC tool loop, prompts, and grading remain unchanged. However, because we did not run an open-weight model, we do not claim that the observed tier interaction extends to one. If the reviewer has a specific open-weight model they consider most informative, please identify it; we will add the adapter, run it on rented GPU compute rather than a closed-model API, and release the complete per-item results.
