# PHREEQC-MCQ-200 Rebuttal Response Tracker

This file tracks one concise response per formal review. It is an internal source for the final rebuttal, not the final submission text.

Status labels: **done** = supported by a checked artifact or existing analysis; **pending** = requires a new run or repository update; **limitation** = cannot be completed responsibly during rebuttal.

## Meta Review (Area Chair wK2b)

**Artifact reproducibility — partially done.** We agree that the submitted anonymous artifact was incomplete. The two full 200-item Gemini 3 Flash agent runs were completed on May 3, 2026 but accidentally omitted during packaging; we have now added their summaries, per-item results, workspaces, and logs. We also fixed the dataset/result path mismatch in the analysis entry point and documented the artifact update. We should only claim end-to-end reproduction of Tables 2, 3, and 5 after every referenced condition has been restored and the clean-clone command has been verified.

**Numeric/free-response evaluation — pending.** We constructed a 182-item numeric-answer version of the benchmark, using simulator-derived targets and numeric tolerances instead of answer options. Existing raw agent artifacts are incomplete, so the minimal defensible plan is to rerun the selected GPT conditions and release all per-item outputs before reporting the comparison. This directly tests whether the tool/no-tool gap survives removal of MCQ distractors.

**Validation and human baseline — limitation.** The dataset was authored and checked by one aqueous-geochemistry expert, not by multiple independent experts. We cannot responsibly add an independent-expert agreement statistic or human baseline during rebuttal. We will correct “expert-validated”/plural-reviewer wording to “expert-authored and simulator-verified,” document the expert’s role and credentials, report the executable uniqueness checks, and state the missing independent validation and human baseline explicitly.

## Reviewer 3hnF

**Single simulator and generality.** We agree that the evidence supports a controlled PHREEQC case study rather than a universal claim about all scientific tools. We will narrow the title/claims and position the durable contribution as a deterministic measurement instrument: exact simulator-derived grading, item-level kept/gained/lost analysis, protocol sensitivity, rerun stability, and trajectory failure diagnosis. A second simulator would require new domain-expert item construction and validation and cannot be added responsibly during rebuttal.

**Relation to prior work.** The finding that tools can help is not itself new. What the controlled deterministic setting adds is attribution: it separates net gain from loss of previously correct items, protocol sensitivity from run-to-run variance, and simulator execution failure from navigation/commit failure on identical executable tasks. We will make this contrast to ScienceAgentBench and ToRA explicit.

**MCQ concern.** The 182-item numeric-answer subset removes answer labels and distractors while preserving the underlying simulator questions. We will report only newly verified runs with released per-item artifacts; the key test is no-tool versus PHREEQC-agent performance, not exact agreement with every MCQ percentage. **Status: pending selected GPT reruns.**

**Effective size and reuse.** The 200 questions come from 21 scenario/setup families and often query different output fields or time steps from the same underlying scenario. Each benchmark item is evaluated in a separate isolated agent session, so agents cannot reuse a simulation across items. We will state both facts prominently rather than presenting 200 as 200 unrelated simulator setups.

**Human calibration and tier labels.** No independent human baseline is available; this will be stated as a limitation. We will formally define tier labels as empirical groupings induced by the observed protocol/failure patterns, not intrinsic rankings of model families.

## Reviewer MmtE

**Numeric/free-response subset.** The 182-item numeric subset directly tests scientific value recovery before A/B/C/D mapping. We will release its prompts, targets/tolerances, per-item predictions, and exact/tolerance-based grading. **Status: dataset and no-tool runs exist; selected agent runs must be rerun and verified.**

**Harness and 24-step budget.** We agree that harness choices are part of the measured compound system. The existing all-model trajectory analysis already separates the mechanisms: 0% of GPT-5.1 and 9.7% of GPT-5.2 failures reach the ceiling, versus 62–68% for top-tier failures, while Gemini’s TOC regression is explicitly a navigation/commit-rate effect (median 9 TOC reads versus 2 under Raw100k). We will avoid presenting the 24-step result as model-only capability and will state that a larger-budget ablation remains future work.

**Uneven rerun coverage.** The current paper has three TOC reruns per top/mid-tier model but less repeat evidence for Raw100k/Raw500k and external harnesses. Under the rebuttal budget, we will prioritize Raw100k repeats for GPT-5.1 and GPT-5.2, release the complete runs, and label all unrepeated comparisons as single-run evidence. **Status: pending.**

## Reviewer h7Cb

**Missing artifacts and failing command.** The reviewer’s report is correct. The anonymous package omitted important agent artifacts and contained inconsistent `dataset` and `result(s)` paths. We have added the two complete Gemini 3 Flash runs and an artifact-update manifest, and fixed the analysis entry point so it runs from the repository root. We will not imply that these files were present during review: the original run metadata/logs record May 3, 2026 execution, while the repository commit records the later packaging correction. **Remaining requirement:** restore or clearly account for every condition used by Tables 2, 3, and 5 before claiming end-to-end table reproduction.

**Expert validation.** One aqueous-geochemistry expert authored and simulator-checked the items. There was no independent expert panel, agreement rate, or human baseline. We will correct the manuscript and Croissant/README metadata accordingly, describe the executable acceptance criteria (successful frozen-version run, unique target, plausible distractors, and question clarity), and state independent validation as future work.

**Scope, version stability, and open-weight models.** We will narrow the framing to a single-simulator case study. Ground-truth claims apply to the pinned PHREEQC version and bundled database; point-release/database invariance has not been tested and will be listed as a limitation. Open-weight models were excluded by the finite API/compute budget, so the observed capability-tier interaction should not be claimed to cover them.

## Submission Checklist

- [x] Add both complete Gemini 3 Flash agent runs to the anonymous artifact.
- [x] Record original run timestamps separately from the later artifact-update commit.
- [x] Verify the analysis command from the anonymous repository root and record its generated files.
- [ ] Account for every artifact underlying Tables 2, 3, and 5; do not claim full reproduction before this is complete.
- [ ] Rerun and release the minimal selected GPT numeric-answer and Raw100k-repeat conditions.
- [ ] Replace plural/independent expert-validation claims in the paper and metadata.
- [ ] Narrow title/generalization language and define capability tiers formally.
