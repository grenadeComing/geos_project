# PHREEQC human evaluation: answer-key review and offline worksheet

This folder holds two things:

1. **An answer-key recheck for the benchmark author** (`for_Sachit.html`, plus the read-only summary `author_review.html`). Every question in the 200-item set was re-run in PHREEQC and sorted into *likely wrong key*, *suspicious*, *could not verify*, or *verified*. The recheck page groups the 42 non-verified questions and 7 scenario setups by cause; the author records decisions and sends back a JSON file.
2. **A human worksheet** (`for_Jerry.html`) of 75 questions: 5 per scenario × 15 scenarios, drawn only from the *verified* items. Five scenarios are on hold until the author fixes them (see below).

## Files

| File | Who it is for | Contents |
|---|---|---|
| `for_Jerry.html` | participants (Jerry) | Offline worksheet. Questions and options only, no answers. |
| `phreeqc.dat` | participants | The frozen database used in all model experiments (copied unchanged). |
| `for_Sachit.html` | benchmark author (Sachit) | Offline page grouping the flagged questions and setups by cause, with one decision per group (keep, change key, revise, drop), per-question overrides, and input-file attachments. Downloads a JSON file. Contains answer keys. |
| `author_review.html` | benchmark author | Read-only summary of the verification results. Contains answer keys. |
| `verification.json` | organizer | Machine-readable verification results for all 200 questions. Contains answer keys. |
| `verification_inputs/` | organizer / author | The PHREEQC input files used for verification, one folder per scenario. |
| `selection_manifest.json` | organizer | Sampling seed, eligible pool, selected IDs and hashes. |
| `scenarios.json`, `template.html`, `style.css`, `app.js` | build sources | Scenario definitions and worksheet code. |
| `recheck_app.js`, `recheck.css` | build sources | Author recheck page code. |
| `build.js`, `author_report.js`, `author_recheck.js`, `grade.js`, `tests/test.js` | organizer | Build, author pages, grading and tests. |

## What to send to the author

Send `for_Sachit.html`. It groups the 42 non-verified questions and 7 scenario setups by cause (wrong key, missing question sentence, moles vs. molality, unstable redox values, S16 location wording, S17 units, S01 step numbering, single issues, setup text), so the author makes about 17 decisions instead of 49. A group decision applies to all its questions, and any question can be overridden. Every question is shown in full (setup, question, A–D with the key marked) above our PHREEQC value and finding. Each scenario shows which of the author's `testN` files it came from, and our inputs have a Copy button. Progress saves in the browser; **Download my review** produces a JSON file that can also be restored into the page later.

The returned JSON lists, per question, the effective decision (`keep`, `change_key`, `revise`, `drop`), whether it came from the group or the question, the new key or revised text and options, and a comment; per scenario setup, whether to revise it and the new text; the group decisions; and any attached PHREEQC input files. Group membership is defined in `author_recheck.js`, which checks that every non-verified question is in exactly one group.

The page shows only PHREEQC evidence. Model results are deliberately left out so the key is corrected independently of the systems being evaluated.

## What to send to participants

The worksheet is the same for every participant (each enters their own participant code). For another evaluator, copy `for_Jerry.html` to `for_<name>.html`; no rebuild is needed.

Send **only** `for_Jerry.html` and `phreeqc.dat`, together with your own PHREEQC 3.8.6 installation instructions and consent paperwork. The worksheet is not a substitute for informed consent or an ethics determination.

Do **not** send the repository, the JSONL datasets, `verification.json`, `author_review.html`, `selection_manifest.json`, or model results. They all contain answers.

Tell participants explicitly to use the provided `phreeqc.dat`. The `phreeqc.dat` that ships with PHREEQC 3.8.6 is a different file (different SHA-256) and gives different numbers for some questions.

## How the answer key was verified

- Engine: PHREEQC 3.8.6 with the frozen `phreeqc.dat` (SHA-256 recorded in `verification.json`).
- S01–S10: Sachit's original input files (`LLM Questions/LLM_Questions_SC/test1–10/Answers+Code.txt`). They come from an earlier version of the questions with different options, but define the setup.
- S11–S20: no original inputs were available, so each scenario was set up from the most literal reading of the question text with PHREEQC defaults (mmol/kgw, 25 °C, pe 4 unless stated).
- Inputs were never tuned to hit the key. Where the literal setup matched no option, a few documented alternative readings were tried and recorded.
- Redox-sensitive values in poorly poised systems were re-run with tighter convergence settings. Values that changed at the digit separating the options are marked *unstable*.
- Statuses: `verified` (keyed option reproduced exactly and uniquely), `wrong_key` (a different option is reproduced), `no_match`, `unstable`, `ambiguous` (defective or ambiguous text), `not_verified` (setup could not be reproduced).
- Model results played no part in any status or in sampling, and are deliberately left out of the author report so the key is corrected on PHREEQC evidence alone. (`verification.json` keeps per-item model agreement for organizer analysis only.)

"Verified" means reproducible under this setup. It is not an independent scientific validation of the question.

## Scenarios on hold

S01, S06, S07, S09 and S17 are not in this worksheet version (`"hold"` in `scenarios.json`):

- **S01**: only 2 verified items; the key's reaction-step numbering is inconsistent.
- **S06, S07**: the text says "pH 3" / "pH 5", but the key needs a charge-balanced starting solution (`pH 3 charge`).
- **S09**: the text says "10 moles" of each gas, but the key needs those numbers as GAS_PHASE partial pressures.
- **S17**: concentrations have no units; the key needs mol/kgw.

When the author sends a revised dataset, re-verify it, remove the holds that are resolved, and build a new study version (new seed and study ID).

## Fixed sampling

- Source: `datasets/neurips_dataset_200_clean.jsonl`, unchanged.
- Eligible pool: questions with status `verified` in `verification.json`, in scenarios not on hold (132 items across 15 scenarios).
- Seed: `phreeqc-human-100-v2-20260930`.
- For each eligible question compute SHA-256(`seed|scenario_id|question_id`), take the 5 smallest per scenario, and show them in original ID order.
- Answer letters and model outcomes do not enter the ranking.
- Question wording and options A–D are shown exactly as in the dataset. A fifth choice, **E = I don't know**, is added as an abstention. It is not a new scientific answer and is not the same as an original "N/A" option.
- S13 merges Q116–124 and Q125 (same sulfuric-acid setup; Q125's preamble adds "to solution 1" and is shown to participants). Scenario labels S14–S20 therefore differ from the older 21-stem analysis. Always join results by original question ID.

If the author revises questions, do not overwrite the old version. Save a new dataset version, re-run verification, bump the seed and study ID, rebuild, and compare only results on the same question version.

## Running the study

1. Give each participant their own browser profile and an anonymous code (P01, P02, …). Names and emails must not be entered.
2. Allowed: PHREEQC, its documentation, a calculator. Not allowed: LLM assistants, answer keys, discussion with other participants. Reusing one's own simulation within a scenario is allowed and can be recorded per scenario.
3. A–D are the original options. E ("I don't know") is recorded separately and scored as incorrect.
4. If a participant thinks a question is flawed, they still choose A–E and switch on **Report a problem** with a reason (missing information, no correct option, more than one correct option, unclear wording, other) and an optional note. A reason is required before finishing. Reports do not change the score; `grade.js` lists them under `problem_reports`.
5. Progress is saved in the browser's localStorage. `file://` storage behaviour differs between browsers; moving the file, private windows or clearing site data can lose progress. Participants should download JSON backups regularly. Restoring a backup asks for confirmation and checks question IDs and version hashes.
6. Finishing requires a participant code and an answer (A–E) for every question. Finalizing locks the page and downloads a JSON file. Nothing is uploaded; the download can be repeated.
7. Participants send the final JSON plus their per-scenario input/output folders to the organizer. Timestamps are not a reliable measure of working time, and the local lock is not tamper-proof.
8. Keep each participant's first final submission. Release answers only after all submissions are in.

## Grading

From the repository root:

    node human_evaluation/grade.js /path/to/final-response.json

The grader reads the answer key locally, never writes it into the worksheet, checks hashes, IDs, scenario assignment and completeness, and prints overall accuracy, the number of E answers, and per-scenario results.

Re-aggregate model results on the same 75 question IDs before comparing with humans. Report that humans may reuse one simulation per scenario while the agents solved each question independently, so this is a human calibration rather than a matched time/cost comparison. Scenarios are equally weighted here, unlike the original 200 items, and five scenarios are absent. Statistical analysis should account for correlation within scenarios and for the same items being answered by several participants.

## Rebuilding and testing

    node human_evaluation/build.js          # worksheet, manifest, database copy
    node human_evaluation/author_report.js  # author_review.html
    node human_evaluation/author_recheck.js # for_Sachit.html
    node human_evaluation/tests/test.js

The browser tests need Playwright: point `NODE_PATH` at an installed Playwright `node_modules`, and optionally set `PLAYWRIGHT_CHROMIUM_EXECUTABLE` to an existing Chromium. The worksheet itself has no dependencies.

Rebuilding overwrites `for_Jerry.html`, `selection_manifest.json`, `author_review.html`, `for_Sachit.html` and the database copy in this folder. It never modifies the source dataset or the paper. Once the worksheet has been sent out, any change to sampling or questions needs a new study ID and seed, and the old files should be kept.
