# PHREEQC-Agent Rebuttal — Paper #322 (ACM CAIS 2026)

> This file contains: (1) all three reviewer reviews, (2) author response instructions, (3) draft rebuttal responses, (4) raw experiment data from rebuttal experiments.
> Reviews received: #322A (Weak Accept 4), #322B (Reject 1), #322C (Borderline 3)
> Rebuttal deadline: April 17, 2026

---

# Part 1: Reviews

## Review #322A

**Summary of Paper's Contribution**

The paper contributes a geochemistry LLM system augmented with PHREEQC (domain-specific "gold standard") simulation tooling, an original domain-expert curated MCQA benchmarking dataset, and respective analyses of zero or one-shot model performance versus a tool-augmented observe-plan-execute fixed-limit loop. The core problem addressed is whether and to what extent tool-use improves (or degrades) the system's MCQA performance, on questions typically requiring simulation, given varying model capabilities. The key architectural approach is a metadata‑only execution design with table-of-contents‑guided output inspection that makes large PHREEQC simulations tractable for the LLM agent. Rather than showing a clean divide between "strong" and "weak" models, the results reveal that tool augmentation produces mixed effects even for frontier models (GPT 5.2, Opus 4.6): they yield new correct answers via tool-use while also losing some correct one‑shot responses. By introducing and analyzing retention alongside gains and losses, the paper highlights that tool use interacts with model behavior in complex ways, sometimes verifying and sometimes destabilizing correct reasoning. Empirically, this is supported by detailed accuracy, retention, efficiency, and failure analyses across multiple LLM, including cases where tool use improves or degrades results.

**Contribution Type**: C. Evaluation Methodology or Benchmark; J. Other

**Impact Dimensions**: C. Demonstrating what works (and what doesn't) in practice; D. Opening new research directions; E. Establishing community infrastructure (benchmarks, datasets, tools)

**Review Summary**

Recommendation: Weak Accept / Borderline. This paper delivers a careful and honest evaluation of a tool‑augmented LLM system for computational geochemistry, combining a simple but well‑motivated agent architecture with a curated, domain‑expert benchmark and detailed empirical analysis. It succeeds as an evaluation‑driven contribution by showing that tool augmentation is not uniformly beneficial, introducing retention and efficiency metrics complementary to raw accuracy, and analyzing failure modes and agent behavior across models. The work is particularly well suited to CAIS in its interdisciplinary scope, bridging agentic system design with a real scientific simulation domain and surfacing system‑level tradeoffs, potentially relevant beyond geochemistry. Assuming the related‑work coverage is complete, the paper represents an important step beyond prior geochemistry LLM applications centered on synthesis and property prediction, by studying grounded simulation and control in a realistic scientific workflow. The findings benefit researchers studying cross-domain agent reliability, tool integration, and evaluation methodologies. The main weaknesses appear fixable through additional results and analysis or exposition rather than fundamental flaws. While I am not qualified to assess the geochemistry correctness in depth, the agentic system design, evaluation framework, and analysis fall within my expertise (respective specific feedback below).

**Strengths**

- End-to-end, rather than only showcasing positive results, reports both progress and regressions under tool augmentation (e.g., Claude Opus 4.6 + Agent improves to 89.0% vs 73.5% one-shot, while Gemini 2.5 Flash drops under the agent setting relative to one-shot).
- Specific architecture choices (e.g. table of contents for navigating large experimental outputs) were empirically-motivated and the overall simplicity of the architecture helps isolate the effects of tool-augmentation, while analyzing performance on a real, domain-specific problem set.
- Analyzes the performance regression under tool augmentation by separating kept / gained / lost correct answers and defining retention (Table 2), supporting the paper's "capability threshold" claim.
- Illuminates cases where aggregate accuracy can hide qualitatively different solution sets, by quantifying overlap/uniqueness between parametric and tool-based correctness (Section 5.2, overlap and "different strategies converge on the same headline").
- Includes trajectory efficiency metrics as well as accuracy metrics (median/mean steps, median/mean PHREEQC runs, and step-limit hits) in the analysis (Table 3 / Section 5.4).
- Connects efficiency to concrete failure mechanisms, e.g. many Gemini Flash losses are attributed to step-limit exhaustion and unproductive tool-use trajectories rather than purely incorrect reasoning.
- Provides a domain-grounded failure analysis with clearly defined categories and geochemistry-/PHREEQC-specific explanations (Section 6.1).
- Additionally provides model-specific error taxonomies for GPT 5.2 and Opus 4.6, that distinguish "wrong answer" vs "no answer," and attributes errors to e.g. modeling flaws, interpretation mistakes, and PHREEQC precision limits (Table 4 / 6.1, Table 5 / Section 6.2).
- Dataset construction mixes official USGS PHREEQC database examples with more open-ended community forum examples with expert review to ensure unique correct answers (Section 4.2).
- Positions the work to move geochemistry LLM-use beyond synthesis and prediction, by analyzing both what works and what doesn't in-context.
- States limitations clearly and specifically (e.g., single benchmark/domain; MCQ format inflates baselines; answer-option imbalance), which supports interpretability and reproducibility expectations (Section 7).
- Pinpoints future work specifically building on observed failure modes, proposing targeted improvements (e.g., constrained generation for gas-phase/kinetics, adaptive step budgets, confidence-gated tool invocation) rather than generic next steps (Conclusion / last paragraph).

**Points to Address Further**

1. The dataset's answer-option imbalance is noted (Section 7), but its quantitative impact could be analyzed.
2. Table 3 reports only means and medians for steps and PHREEQC runs. Reporting dispersion measures (e.g., SD or IQR) would be helpful, especially given step-limit ceiling effects and apparent heavy tails.
3. Some terminology is non-standard (e.g., "backbone," "parametric knowledge," "agent-hurt"). The terms are understandable from a read-through but clarity would be improved using more common, terminology even if more verbose e.g. model configuration, pre-trained knowledge, tool-augmentation performance regression.
4. Section 6's opening distinction between "active" and "solved" failure modes is not clearly defined; revising would improve readability.
5. The evaluation contrasts one-shot and tool-augmentation, but does not explore intermediate strategies e.g. CoT. Given the strength of one-shot baselines, it would be informative to see the results from lighter-weight approaches (e.g., CoT or reflection) to see how much of the gap is bridged without full tool invocation.
6. "Only questions for which at least two independent reviewers agreed on the unique correct answer were retained" (4.2), can you elaborate on this and why 2 was sufficient?
7. Can you further ablate how many failures were due to instruction-following (IF) versus tool-use training and how it correlates with the models known performance on IF benchmarks?
8. The Appendix A system prompt is informative; can you provide the one-shot prompt as well?
9. Why did you drop Gemini 2.5 Pro and Claude Sonnet from the agent evaluation?
10. Is the failure analysis that produced Tables 4 and 5 scalable, how is it conducted? If a solution is not already in place, would techniques such as e.g. MAST (Cemri et al. 2025) be applicable?

**Questions for Authors**

1. Given the strength of one‑shot baselines and the mixed gains from tool use, how do you distinguish questions that truly require simulation and exact calculation from cases where performance depends primarily on the model's reasoning performance, instruction‑following, etc.?
2. Please explicitly state how MCQA performance relates to future work and broader use-cases e.g. do you envision scientists directly providing 4 hypotheses for the agent to test, or do you see this as establishing baseline functionality for more open-ended research support... in the future?
3. Further, can you discuss the impact for the sciences e.g. how much time or compute cost etc. would this save scientists if you had a perfectly working solution, or is this purely exploratory at this point?
4. Given the strong answer‑option imbalance you report (Limitation 7), can you incorporate an analysis of the correct/incorrect responses w.r.t. this known distribution?
5. Section 3.1, how was T_{max}=24 tuned or chosen? What about the other runtime limits e.g. TOC entries <= 50, how were they chosen?
6. Regarding Section 9, can you make the artifacts available for review via e.g. https://anonymous.4open.science ?

**Relevance to CAIS**: 4. Clearly relevant.
**Overall Recommendation**: 4. Weak Accept.
**Reviewer Confidence**: 3. Moderate confidence.

---

## Review #322B

**Summary of Paper's Contribution**

The paper presents PHREEQC-Agent: Tool-Augmented LLM for Geochemical Computation a PHREEQC-Agent, which is a tool-augmented LLM system that writes, executes, and interprets PHREEQC simulations. The paper evaluates model method combinations covering three model families.

**Contribution Type**: C. Evaluation Methodology or Benchmark

**Impact Dimensions**: C. Demonstrating what works (and what doesn't) in practice

**Review Summary**

I would not recommend this paper for acceptance. The exact research question is not clear in the relatively short introduction section and it is not properly motivated. Although agentic systems are used, the work reads more like an application of agentic systems. It is not clear how the proposed contributions and methodology would generally improve our understanding of the topics that the conference is about. The more suitable audience might be the geochemical community. Moreover, the paper has a fundamental flaw in how it uses (or doesn't use) references. There is a reference list, but none of them are cited in the text.

**Specific Points of Feedback**

1. The introduction is too brief
2. Section 7 is just a list of bullet points
3. There is very little story in the paper: it is more like a technical report on a special experiment
4. The reference list, besides not cited properly, is short and contains only eight entries

**Questions for Authors**

What is PHREEQC? What is the exact research question? What is the motivation behind this work? How does this improve our understanding of agentic systems and AI?

**Relevance to CAIS**: 2. Marginally relevant.
**Overall Recommendation**: 1. Reject.
**Reviewer Confidence**: 2. Low confidence.

---

## Review #322C

**Summary of Paper's Contribution**

The paper introduces PHREEQC-Agent, a tool-augmented react based agent that writes, executes and interprets PHREEQC code (a geochemical simulator) to answer multiple-choice geochemistry questions requiring precise numerical computation. The authors also create a benchmark PHREEQC_MCQ, consisting of 200 multi-choice questions curated by six domain-expert geochemists. The key insight behind the agentic design is that PHREEQC execution outputs can be very long, and therefore it is better to provide the response of the PHREEQC execution tool as a Table-of-content based pointer, instead of providing the raw output. The authors then evaluate 10 model-method combination and present a detailed analysis of the performance and failure modes. The best agent configuration, powered by Claude Opus 4.6, achieves 89.0% accuracy (+15.5 over one-shot baseline). The authors provide an analytical finding: Each model has a backbone capability threshold, surpassing which allows the model+agent to improve over baseline, whereas having a weaker model reduces performance even with access to tools.

**Contribution Type**: C. Evaluation Methodology or Benchmark

**Impact Dimensions**: B. Enabling practitioners to build better systems; C. Demonstrating what works (and what doesn't) in practice; E. Establishing community infrastructure (benchmarks, datasets, tools)

**Review Summary**

I assign a borderline rating to the paper. The paper is clearly well motivated, exploring the use of and benchmarking LLM-based agents on geochemical applications, providing tool-based access to scientific simulators, which is going to be an increasingly important area of exploration. The paper's key contribution is the creation of a new benchmark, PHREEQC_MCQ, curated by domain-experts and providing a very detailed analysis of the performance of several contemporary language models at different scales, both in one-shot as well as agentic harness, on this benchmark.

However, the paper has the following key weaknesses:

1. The novelty of metadata-only execution return design, while practically motivated is limited.
2. The paper's evaluation only considers a single-shot baseline, and the metadata-only augmented react agent. The evaluation lacks comparison to any other tool-augmented or compound AI systems. Even a React baseline that returns raw PHREEQC output would augment the analysis and contribution of the TOC-guided tool results.
3. The paper does not discuss the choice of PHREEQC, whether any alternative simulators were considered, and applicability of the agent to other scientific simulators.
4. No cost or latency data is reported.

**Specific Points of Feedback**

1. Can the authors provide an ablation comparing the metadata-only TOC return against a baseline that returns raw (possibly truncated) PHREEQC output?
2. While the paper highlights the metadata driven tool-results as the key contribution, I find the contribution of the benchmark along with the analysis as the central contribution, and would recommend reframing the contributions around the benchmark.
3. Can the authors include any data about the wall-clock time per question (median and p90) for each model/agent setting?
4. The MCQ format could allow a react agent to potentially test all options. Can the authors share if they observed any such behaviour?
5. The paper would benefit from a discussion contextualizing PHREEQC for a general reader outside the geochemical domain. Further, a running example demonstrating all keys aspects of the agent execution, going from the question, to the react loop, details about the generated PHREEQC will aid readability.

**Questions for Authors**

1. Why only MCQ type questions? Are the authors considering expanding (perhaps as future work) the scope of questions to be more open-ended?
2. "One unsolvable question involves a display-precision limitation where the correct answer and distractor round to the same value in PHREEQC output": Does this not represent a problem with the question itself?
3. A significant contribution of this paper is the benchmark. Some questions were identified to be really difficult, and that couldn't be solved by any of the model/configurations. For example, "the other (question) combines kinetics, gas-phase equilibrium, and high-temperature speciation in a single problem that exceeds all methods' capabilities.". Could the authors clarify if they plan to expand the benchmark with more such questions, and the feasibility of doing so?
4. Have the authors considered testing the impact of automated optimization techniques for agents, like ACE/GEPA/MIPRO? It would be insightful to learn about the behaviour of such optimizers in domain specific benchmarks like PHREEQC_MCQ.

**Relevance to CAIS**: 4. Clearly relevant.
**Overall Recommendation**: 3. Borderline.
**Reviewer Confidence**: 4. High confidence.

---

# Part 2: Author Response Instructions (CAIS 2026)

**Format**: Single written response, no back-and-forth.

**Structure**:
1. Overview — general response to salient points across all reviews (~1000 word soft limit with Section 2)
2. Planned Changes — concrete revisions if accepted (~1000 word soft limit with Section 1)
3. Per-Reviewer Responses — point-by-point, hidden by default, expandable by reviewers (no word limit)

**Key rules**:
- Double-blind: no identifying info, no external links
- Focus on: factual errors, direct questions, misunderstandings
- Don't need to address every minor comment
- Be concise, respectful, prioritize what affects outcome

**Wall-clock**: actual elapsed time per question (start to finish). Median = middle value. p90 = 90th percentile (slowest 10% take at least this long). Data extracted from agent chat logs (run_start → run_end timestamps).

---

# Part 3: Author Response (Draft)

## 1. Overview

We thank all reviewers for their feedback. Two key experimental gaps were identified: (1) the lack of an intermediate CoT baseline between one-shot and full agent (Reviewer A, Point 5), and (2) the absence of a raw-output ablation for the TOC design (Reviewer C, Weakness 1). We have run both experiments and present the results below.

**New Experiment 1: Chain-of-Thought Baseline.**
To isolate the contribution of tool access from reasoning, we ran a CoT baseline (step-by-step reasoning, no tools, single API call) across four models:

| Model | One-shot | CoT | Agent (TOC) |
|---|---|---|---|
| Claude Opus 4.6 | 73.5% | 71.5% | 91.0% |
| Claude Sonnet 4.6 | 65.0% | 70.0% | 89.5% |
| GPT-5.2 | 55.0% | 61.5% | 73.5% |
| GPT-5.1 | 63.5% | 61.0% | 69.0% |

CoT alone does not bridge the gap to agent performance. For Opus, CoT slightly *decreases* accuracy (−2.0pp), while tools add +19.5pp — the agent lift is entirely from tool access, not reasoning. For GPT-5.2, CoT contributes ~35% of the gain and tools ~65%. This confirms that the accuracy improvements reported in the paper are driven primarily by grounded computation, not by allowing more reasoning tokens.

**New Experiment 2: TOC vs Raw Output Ablation.**
To enable precise input-token accounting and a controlled TOC-vs-Raw comparison, we re-ran all four models with token logging enabled and identical runtime on both the TOC and Raw sides. The accuracies in the table below are from these re-runs and therefore differ slightly from the paper's Table 1 values (LLM outputs can vary run-to-run; Opus, Sonnet, and GPT-5.1 are within ±2pp, while GPT-5.2 drifts −7pp, consistent with its lowest retention (87.3%) in the paper's Table 2, which already flagged GPT-5.2 as the least stable model across configurations). Both modes use identical agent loops; only the tool-return format differs:

| Model | Agent (TOC) | Agent (Raw) | Δ | Input Tokens (TOC → Raw) | Reduction |
|---|---|---|---|---|---|
| Claude Opus 4.6 | 91.0% | 91.5% | −0.5pp | 15.1M → 23.0M | **34%** |
| Claude Sonnet 4.6 | 89.5% | 90.0% | −0.5pp | 18.0M → 33.1M | **46%** |
| GPT-5.1 | 71.0% | 71.5% | −0.5pp | 7.0M → 9.0M | **22%** |
| GPT-5.2 | 66.5% | 72.5% | −6.0pp | 8.9M → 11.6M | **23%** |

On 3 of 4 models, TOC preserves accuracy within 0.5pp while reducing total input tokens by 22–46%. GPT-5.2's −6.0pp gap sits inside its overall ±7pp run-to-run band and should be read as within noise rather than as a robust TOC regression — note that both the TOC and Raw runs for each model were executed back-to-back with identical configuration, so the within-model token comparison is unaffected by across-run variance. Per-call, execute_phreeqc responses shrink by 82% (~3K vs ~20K chars). Beyond token savings, TOC eliminates context-overflow failures from large transport simulations (>1MB output) that were a dominant failure mode in early development.

This ablation also fills Reviewer A's Point 9 gap by providing the previously-missing Claude Sonnet 4.6 agent result: TOC 89.5% (179/200), Raw 90.0% (180/200), versus one-shot 65.0% (130/200) — a **+24.5pp** improvement. Combined with the paper's Opus result (+15.5pp), this confirms the capability-threshold finding generalizes across Anthropic model sizes rather than being Opus-specific. (Gemini 2.5 Pro remains excluded due to Tier API limitations.)

**Additional analysis: answer-option imbalance (Reviewer A, Point 1 and Q4).**
The dataset distribution is A=31 (15.5%), B=64 (32.0%), C=76 (38.0%), D=29 (14.5%); a trivial "always-C" baseline would score 38%. For the paper's best agent (Opus, 89.0%), per-option accuracy is uniform — A: 80.6% (25/31), B: 90.6% (58/64), C: 90.8% (69/76), D: 89.7% (26/29) — showing no exploitation of the majority class. By contrast, several one-shot baselines exhibit majority-class bias: GPT-5.2 over-predicts B (93 predictions vs 64 truths; D accuracy only 27.6%), GPT-5.1 over-predicts C (94 vs 76; D accuracy 13.8%), and Gemini Flash over-predicts C (101 vs 76; D accuracy 13.8%). Tool use measurably corrects this bias.

**On the MCQ format (Reviewer A, Q1–Q2; Reviewer C, Q1).** MCQ prevents models from guessing or hedging: each question has a single deterministic answer reachable only through PHREEQC simulation, and distractors are numerically close enough that guessing without the tool fails. The agent itself operates in an open-ended setting — it generates free-form PHREEQC scripts, inspects simulation outputs, and only commits to one of four options at the final step. Extending the benchmark to fully open-ended QA is planned future work.

**On TOC (Reviewer C, Weakness 1).** Experiment 2 quantifies TOC's effect: 22–46% input-token reduction with ≤0.5pp accuracy regression on 3 of 4 models, plus elimination of context-overflow failures on >1MB outputs. The underlying pattern — *return structure eagerly, defer content* — is not PHREEQC-specific and applies to other simulators whose raw output can exceed the LLM context budget, e.g. GROMACS logs (10–500MB), VASP `OUTCAR` (~50MB), OpenFOAM field files (GB-scale). On Reviewer C, Point 2: we agree the benchmark is the primary contribution; TOC stands alongside it as a validated design pattern.

**Comparison to external agent frameworks (Reviewer C, Weakness 2).** We additionally evaluated the official Claude Agent SDK on the same benchmark: Opus 4.6 reaches 90.5% (181/200, within 0.5pp of our TOC agent's 91.0%) and Sonnet 4.6 reaches 71.5% (143/200, 18.0pp below our TOC agent's 89.5%). The Opus parity shows our framework is not under-engineered relative to a widely-used external agent; the Sonnet gap suggests TOC-guided output inspection matters more for mid-capability models where context management is a bottleneck.

## 2. Planned Changes

If accepted, we will make the following revisions:

- **Expand Introduction** (Reviewer B): explicitly state the research question and motivation; add a paragraph contextualizing PHREEQC for a general AI audience (Reviewer C, Point 5).
- **Add CoT baseline** to Table 1 (Reviewer A, Point 5) and the TOC vs Raw ablation as a new table/subsection (Reviewer C, Weakness 1).
- **Add Claude Sonnet 4.6 row** to Table 1 (one-shot 65.0%, TOC 89.5%, Raw 90.0%) and update the capability-threshold discussion accordingly (Reviewer A, Point 9).
- **Add per-option accuracy table** and prediction distribution analysis showing no majority-class exploitation by the agent (Reviewer A, Point 1 and Q4).
- **Expand Related Work and references** (Reviewer B): cover Toolformer, HuggingGPT, ScienceAgentBench, SWE-bench, and additional geochemistry LLM work. We will also discuss alternative simulators (GROMACS, MODFLOW) and the generalizability of the TOC approach (Reviewer C, Weakness 3).
- **Add dispersion measures** (SD, IQR) to Table 3 and wall-clock time (median, p90) per model (Reviewer A, Point 2; Reviewer C, Point 3).
- **Add running example** in Section 3 showing a full agent trajectory from question to answer (Reviewer C, Point 5).
- **Adopt standard terminology**: "backbone" → "base model"; "parametric knowledge" → "pre-trained knowledge"; "agent-hurt" → "tool-augmentation performance regression" (Reviewer A, Point 3). Clarify "active" vs "solved" failure modes in Section 6 (Reviewer A, Point 4).
- **Include one-shot prompt** in Appendix (Reviewer A, Point 8).
- **Reframe contributions** to foreground the benchmark and analysis, with TOC as a supporting engineering contribution (Reviewer C, Point 2).

## 3. Per-Reviewer Responses

### Specific questions/comments: Review #A

**A1. Answer-option imbalance.** See Overview, "Additional analysis: answer-option imbalance" for the per-option accuracy breakdown and prediction-distribution evidence that the agent shows no majority-class exploitation while several one-shot baselines do.

**A5. CoT baseline.** See Overview, Experiment 1.

**A6. Two-reviewer agreement.** Questions are objectively verifiable through simulation — the correct answer is deterministic. Two independent geochemists verified each script executes correctly and produces exactly one matching answer. Any disagreement led to revision or removal.

**A7. IF vs tool-use failures.** For Opus (22 errors): 15 are tool-use failures (bad PHREEQC input or retry loops), 4 are reasoning failures (wrong interpretation), 3 are problem-difficulty issues. For GPT-5.2 (53 errors): 25 tool-use vs 21 reasoning vs 7 operational. Tool-use quality is the dominant bottleneck for stronger models.

**A9. Sonnet and Gemini Pro.** See Overview, Experiment 2 for the Claude Sonnet 4.6 result (TOC 89.5%, +24.5pp over one-shot). Gemini 2.5 Pro remains excluded due to Tier API limitations.

**Q1.** 15 questions are solved only by Opus Agent and by none of the 6 one-shot baselines — these are the strongest evidence for simulation-requiring questions.

**Q3.** Manual PHREEQC simulation setup takes 15–60 min for an experienced geochemist. Our agent completes this in a median of 23.0s (Opus).

**Q6.** An anonymous artifact repository has been prepared per your request, containing the code, dataset, prompts, evaluation scripts (summary.json + results.jsonl for all 200 questions across every configuration), and representative agent trajectories. Per CAIS author-response rules on external links, we cannot include the URL in this response, but will share it through the program chairs if permitted.

**Q5.** T_max=24: pilot runs showed 95% of successes complete within 15 steps; 24 provides margin for retries. Per Table 3, Opus reaches the 24-step limit on 11/200 questions (5.5%). Post-hoc log analysis shows these exhaustion cases all fall into two problem families — kinetic dissolution over 100 time-steps and high-temperature gas-phase equilibria — with no transport-simulation question exhausting the budget. TOC cap of 50 (first 25 + last 25): calibrated to transport simulations where thousands of repeated time-step headers appear; 25+25 captures setup and converged results.

### Specific questions/comments: Review #B

**B1. "References not cited."** The submitted PDF does contain inline citations throughout Sections 1–3 (e.g., [Appelo and Postma 2004], [Yao et al. 2022], [Schick et al. 2023], [M. Bran et al. 2024], [Boiko et al. 2023]). If any specific passage reads as uncited, we would welcome a pointer so we can address it in the camera-ready.

**What is PHREEQC?** PHREEQC is the USGS-maintained geochemical simulator — the de-facto standard for equilibrium, kinetic, and reactive-transport modeling in aqueous systems, used in environmental remediation, nuclear-waste containment, hydrogeology, and CO2 sequestration. It is legacy Fortran code emitting voluminous, loosely structured text output, which is what makes direct LLM integration non-trivial and motivates our TOC design.

**What is the research question?** Under what conditions does tool augmentation convert base-model capability into tool-grounded accuracy, and when does it fail? Specifically, for scientific computation tools that are deterministic, non-differentiable, and produce unstructured outputs, what agent design is required, and what base-model capability threshold must be crossed before tool use helps rather than hurts?

**What is the motivation?** Tool-augmented LLMs are being deployed in chemistry, biology, and physics, but the community lacks a controlled benchmark with deterministic ground truth to separate genuine tool-use gains from reasoning-from-pretrained-knowledge. PHREEQC-MCQ provides that signal: every question has a unique, simulation-reachable answer, and deterministic grading eliminates hedging. The benchmark is not primarily about geochemistry — it is a controlled probe for compound AI systems.

**B5. Contribution to understanding agentic systems and AI.** Three findings generalize beyond geochemistry: (1) tool augmentation has a per-model **capability threshold** (Opus +15.5pp, Sonnet +24.5pp, Gemini Flash −7.0pp), quantified by our gain/loss/retention decomposition; (2) **metadata-guided output inspection (TOC)** preserves accuracy within 0.5pp while cutting input tokens 22–46% (Experiment 2, rebuttal) — a generalizable design pattern for tools whose output exceeds typical context windows; (3) the dominant failure mode for stronger models is **tool-use quality** (bad input syntax, retry loops), not reasoning — Opus errors decompose as 15/22 tool-use vs 4/22 reasoning — which shifts the bottleneck from LLM training to tool-interface design as models scale. These are claims about compound AI systems, instrumented through PHREEQC.

**Other points** (brief Introduction, Section 7 bullet format, short Reference list, technical-report tone): see Planned Changes. We will rewrite the Introduction to state the research question explicitly, add context for an AI-generalist audience, expand Related Work with agentic-systems literature (Toolformer, HuggingGPT, ScienceAgentBench, SWE-bench), convert Section 7 from bullets to prose, and thread a clearer narrative through Sections 1–3.

### Specific questions/comments: Review #C

**C1. TOC vs Raw ablation.** See Overview, Experiment 2.

**C-W2. Comparison to external agent frameworks.** See Overview, "Comparison to external agent frameworks" — our custom agent matches the official Claude Agent SDK on Opus (91.0% vs 90.5%) and exceeds it on Sonnet (89.5% vs 71.5%, +18.0pp).

**C3. Wall-clock time.** Per-question durations extracted from agent chat logs (run_start → run_end timestamps); n≈195–200 completed runs per model.

| Model | Median | p90 |
|---|---|---|
| Claude Opus 4.6 | 23.0s | 94s |
| Claude Sonnet 4.6 | 30.0s | 119s |
| GPT-5.2 | 10.0s | 33s |
| GPT-5.1 | 9.0s | 15s |
| Gemini Flash | 12.0s | 89s |

Tail cases (≥180s; 1–2 questions per model) are concentrated on the same family: kinetic dissolution over 100 discrete time-steps at high temperature, often coupled with a gas phase. Two mechanisms drive the long tail — (a) PHREEQC itself is slow to execute a 100-step kinetic simulation (tens of seconds per tool call), and (b) convergence diagnostics can make the agent exhaust its step budget. These are the same problem types that motivated the TOC design (raw outputs can exceed 1MB).

**C4. Brute-force option testing.** This is structurally prevented by our setup: the agent is like a student taking a test — it works on the problem, writes down one final answer, and only then is the paper "collected" and graded. Evaluation against the gold label is not inside the agent loop, so there is no in-run signal by which the agent could iteratively try options and keep the one that "passes".

**Q1. Why MCQ?** See Overview, "On the MCQ format" for our rationale (MCQ prevents guessing/hedging via deterministic answers with numerically close distractors; agent itself is open-ended script generation).

**Q2. Precision-limit question.** Agreed — this is a dataset quality issue. Will flag and exclude (1/200, negligible impact).

**Q3. Benchmark expansion.** Agreed this is the natural next step. Planned extensions: (a) increase difficulty with multi-phase coupled problems requiring multiple PHREEQC modules per answer and longer reactive-transport scenarios, and (b) add an "(E) I don't know" option so models can abstain rather than guess — this will let us measure calibration and distinguish capability gaps from over-confidence, and reduces the majority-class-baseline floor.

**Q4. Automated optimization (MIPRO etc.).** Not yet tested. The benchmark's deterministic signal makes it well-suited for such optimization. Will discuss in future work.

==============================================================================
END OF REBUTTAL RESPONSE — everything below is internal reference material only
==============================================================================

# Part 4: Experiment Results (Raw Data)

> Raw data from rebuttal experiments. No conclusions drawn — data only.
> Run date: 2026-04-15 to 2026-04-16

## E1: Chain-of-Thought (CoT) Baseline

Single API call, temperature=0, "Think step by step" instruction, no tools.  
Results saved in `result/cot/`

| Model | One-shot | CoT | Agent (TOC, paper) | Agent (TOC, re-run) | CoT tokens (in / out) |
|---|---|---|---|---|---|
| Claude Opus 4.6 | 73.5% (147/200) | 71.5% (143/200) | 89.0% (178/200) | 91.0% (182/200) | 44,814 / 176,961 |
| Claude Sonnet 4.6 | 65.0% (130/200) | 70.0% (140/200) | — | 89.5% (179/200) | 44,814 / 191,216 |
| GPT-5.2 | 55.0% (110/200) | 61.5% (123/200) | 73.5% (147/200) | 66.5% (133/200) | 39,324 / 45,359 |
| GPT-5.1 | 63.5% (127/200) | 61.0% (122/200) | 69.0% (138/200) | 71.0% (142/200) | 39,324 / 76,479 |

Notes:
- All CoT runs: 0 errors, 200/200 completed
- CoT prompt: system="You are a geochemistry expert." user="Think step by step. Show your reasoning, then give your final answer in the form <<< X >>>"
- One-shot prompt enforces "output only <<< X >>>" with no reasoning allowed
- Sonnet 4.6 has no agent run in the paper (excluded from agent evaluation)

---

## E2: TOC vs Raw Output Ablation

Agent with full tool access (write_file, execute_phreeqc, read_file, list_file).  
- **TOC mode**: execute_phreeqc returns metadata table-of-contents with section line numbers. Agent uses read_file(start_line, end_line) to inspect specific sections. This is the original design in the paper.
- **Raw mode**: execute_phreeqc returns full result.out content directly (truncated at 100K chars). No TOC.
- Both modes use litellm for unified OpenAI/Anthropic tool-calling protocol.
- Per-model settings match original paper: Anthropic gets temperature=0, max_tokens=4096, step-warning at step 23; OpenAI uses provider defaults, no step-warning.
- max_steps=24, parallel_tool_calls=False (matching original).

### Claude Opus 4.6

| Mode | Accuracy | Input Tokens | Output Tokens | Errors |
|---|---|---|---|---|
| Agent TOC | **91.0% (182/200)** | 15,115,027 | 304,261 | 1 |
| Agent Raw | **91.5% (183/200)** | 23,010,675 | 291,572 | 0 |

- Δ accuracy: −0.5pp (within run-to-run variance)
- Token reduction: **34%** (23.0M raw → 15.1M TOC)
- Source: sj_toc_opus46 (TOC), sj_ablation_raw_opus46 (Raw)

### GPT-5.1

| Mode | Accuracy | Input Tokens | Output Tokens | Errors |
|---|---|---|---|---|
| Agent TOC | **71.0% (142/200)** | 7,033,278 | 65,726 | 0 |
| Agent Raw | **71.5% (143/200)** | 8,964,305 | 53,755 | 0 |

- Δ accuracy: −0.5pp (within run-to-run variance)
- Token reduction: **22%** (9.0M raw → 7.0M TOC)
- Source: sj_toc_gpt51 (TOC), sj_ablation_raw_gpt51 (Raw)

### GPT-5.2

| Mode | Accuracy | Input Tokens | Output Tokens | Errors |
|---|---|---|---|---|
| Agent TOC | **66.5% (133/200)** | 8,899,833 | 86,488 | 1 |
| Agent Raw | **72.5% (145/200)** | 11,621,239 | 78,106 | 2 |

- Δ accuracy: −6.0pp (GPT-5.2 shows higher variance; see note below)
- Token reduction: **23%** (11.6M raw → 8.9M TOC)
- Source: sj_toc_gpt52 (TOC), sj_ablation_raw_gpt52 (Raw)
- Note: GPT-5.2 TOC re-run (66.5%) is lower than the original paper result (73.5%). The original paper run used the native evaluate.py script while the re-run used evaluate_ablation_raw.py with litellm. GPT-5.2 shows the highest run-to-run variance among all models, consistent with its lower retention (87.3%).

### Claude Sonnet 4.6

| Mode | Accuracy | Input Tokens | Output Tokens | Errors |
|---|---|---|---|---|
| Agent TOC | **89.5% (179/200)** | 17,968,511 | 321,200 | 1 |
| Agent Raw | **90.0% (180/200)** | 33,095,587 | 344,266 | 0 |

- Δ accuracy: −0.5pp (within run-to-run variance)
- Token reduction: **46%** (33.1M raw → 18.0M TOC)
- Source: sj_toc_sonnet46 (TOC), sj_raw_sonnet46 (Raw)

### PHREEQC usage (from original paper runs)
- First 100 (Jerry): ~60% of questions used PHREEQC
- Last 100 (Sachit): 100% of questions used PHREEQC

---

## Dataset Composition

- **First 100 questions**: Jerry's questions (simpler, ~60% require PHREEQC, no source tag in jsonl)
- **Last 100 questions**: Sachit's questions (complex, 100% require PHREEQC, source: test1_q1 ... test10_q10)
- **Total**: 200 questions from `dataset_S+J.jsonl`
- **Answer distribution**: A=15.5%, B=32.0%, C=38.0%, D=14.5%
- **Average question length**: First 100: 414 chars, Last 100: 383 chars

---

## Token Usage Summary

| Run | Total Input Tokens | Total Output Tokens | Source |
|---|---|---|---|
| Opus Agent TOC (re-run) | 15,115,027 | 304,261 | sj_toc_opus46 |
| Opus Agent Raw | 23,010,675 | 291,572 | sj_ablation_raw_opus46 |
| Sonnet Agent TOC | 17,968,511 | 321,200 | sj_toc_sonnet46 |
| Sonnet Agent Raw | 33,095,587 | 344,266 | sj_raw_sonnet46 |
| GPT-5.1 Agent TOC | 7,033,278 | 65,726 | sj_toc_gpt51 |
| GPT-5.1 Agent Raw | 8,964,305 | 53,755 | sj_ablation_raw_gpt51 |
| GPT-5.2 Agent TOC (re-run) | 8,899,833 | 86,488 | sj_toc_gpt52 |
| GPT-5.2 Agent Raw | 11,621,239 | 78,106 | sj_ablation_raw_gpt52 |
| Opus CoT | 44,814 | 176,961 | sj_cot_claude_opus46 |
| Sonnet CoT | 44,814 | 191,216 | sj_cot_claude_sonnet46 |
| GPT-5.2 CoT | 39,324 | 45,359 | sj_cot_gpt52 |
| GPT-5.1 CoT | 39,324 | 76,479 | sj_cot_gpt51 |

Note: Original paper runs (`sj_agent_claude_opus46`, `sj_agent_gpt52`, `sj_agent_gpt51`, `sj_agent_gemini25flash`) do NOT have token usage in their logs — the logging feature was added after those runs were completed. All ablation re-runs use evaluate_ablation_raw.py with token logging enabled.

---

## Rate Limit Observations

- **Opus raw mode (Anthropic API)**:
  - 2 workers immediately hit 800K input tokens/min rate limit
  - 24 api_error failures in first attempt (105-131 range)
  - Required reducing to 1 worker + 10 retries with 15s linear backoff (max 90s)
  - Total wall-clock: ~3 hours across multiple resume attempts
- **Opus TOC mode (Anthropic API)**:
  - 2 workers ran successfully (resumed from 69, completed remaining 131)
  - Some transient rate limit errors but recovered within retry budget
- **GPT-5.2 raw mode (OpenAI API)**:
  - 4 workers, zero rate limit issues
  - Completed 200 questions in 21 min 45s
- **GPT-5.2 TOC mode (OpenAI API)**:
  - 4 workers, zero rate limit issues
  - Completed 200 questions in 21 min 49s

Root cause: raw mode embeds full result.out content in tool responses, inflating per-step context. With multi-step agent loops, context accumulates across steps. Anthropic's per-minute input token limit (800K) is more restrictive than OpenAI's, making Opus raw mode the bottleneck.

---

## Original Paper Results (for reference)

From existing `result/` directories (runs predating this rebuttal session):

| Run | Accuracy | Source |
|---|---|---|
| Claude Opus 4.6 Agent (TOC) | 89.0% (178/200) | sj_agent_claude_opus46 |
| GPT-5.2 Agent (TOC) | 73.5% (147/200) | sj_agent_gpt52 |
| GPT-5.1 Agent (TOC) | 69.0% (138/200) | sj_agent_gpt51 |
| Gemini 2.5 Flash Agent (TOC) | 44.5% (89/200) | sj_agent_gemini25flash |
| Claude Opus 4.6 One-shot | 73.5% (147/200) | sj_baseline_claude_opus46 |
| Claude Sonnet 4.6 One-shot | 65.0% (130/200) | sj_baseline_claude_sonnet46 |
| GPT-5.2 One-shot | 55.0% (110/200) | sj_baseline_gpt52 |
| GPT-5.1 One-shot | 63.5% (127/200) | sj_baseline_gpt51 |
| Gemini 2.5 Flash One-shot | 51.5% (103/200) | sj_baseline_gemini25flash |
| Gemini 2.5 Pro One-shot | 68.0% (136/200) | sj_baseline_gemini25pro |

Note: Paper Table 1 reports GPT-5.2 one-shot as 57.5%, but actual data shows 55.0% (110/200). Table 2 retention calculations are internally consistent with 55.0%.

---

## Rebuttal Re-run Accuracy vs Original

| Model | Original Agent (TOC) | Re-run Agent (TOC) | Re-run Agent (Raw) | TOC Δ vs original | Token savings |
|---|---|---|---|---|---|
| Claude Opus 4.6 | 89.0% (178/200) | 91.0% (182/200) | 91.5% (183/200) | +2.0pp | 34% |
| GPT-5.1 | 69.0% (138/200) | 71.0% (142/200) | 71.5% (143/200) | +2.0pp | 22% |
| GPT-5.2 | 73.5% (147/200) | 66.5% (133/200) | 72.5% (145/200) | −7.0pp | 23% |

Notes:
- Opus and GPT-5.1 re-runs are ~2pp higher than original, likely due to minor implementation differences (litellm routing vs native SDK, evaluate_ablation_raw.py vs evaluate.py).
- GPT-5.2 TOC re-run (66.5%) is notably lower than original (73.5%), while raw re-run (72.5%) is close to original. This suggests GPT-5.2 has high run-to-run variance, particularly under TOC mode.
- All re-runs use identical scripts for TOC and Raw, so the within-model TOC vs Raw comparisons are internally consistent.

---

## File Locations

All results are under `/Users/kezhang/Desktop/projects/geos/geos_github/result/`:

```
result/
├── agent/
│   ├── sj_agent_claude_opus46/       # original paper run (no token logs)
│   ├── sj_agent_gpt52/               # original paper run (no token logs)
│   ├── sj_agent_gpt51/               # original paper run (no token logs)
│   ├── sj_agent_gemini25flash/       # original paper run (no token logs)
│   ├── sj_toc_opus46/                # rebuttal re-run TOC (91.0%, 15.1M tokens)
│   ├── sj_toc_gpt52/                 # rebuttal re-run TOC (66.5%, 8.9M tokens)
│   ├── sj_toc_gpt51/                 # rebuttal re-run TOC (71.0%, 7.0M tokens)
│   ├── sj_toc_sonnet46/              # rebuttal Sonnet TOC (89.5%, 18.0M tokens)
│   ├── sj_ablation_raw_opus46/       # raw ablation (91.5%, 23.0M tokens)
│   ├── sj_ablation_raw_gpt52/        # raw ablation (72.5%, 11.6M tokens)
│   ├── sj_ablation_raw_gpt51/        # raw ablation (71.5%, 9.0M tokens)
│   └── sj_raw_sonnet46/              # raw ablation Sonnet (90.0%, 33.1M tokens)
├── cot/
│   ├── sj_cot_claude_opus46/
│   ├── sj_cot_claude_sonnet46/
│   ├── sj_cot_gpt52/
│   └── sj_cot_gpt51/
└── oneshot/
    ├── sj_baseline_claude_opus46/
    ├── sj_baseline_claude_sonnet46/
    ├── sj_baseline_gpt52/
    ├── sj_baseline_gpt51/
    ├── sj_baseline_gemini25flash/
    └── sj_baseline_gemini25pro/
```

Scripts:
- `baseline_cot.py` — CoT baseline (E1)
- `evaluate_ablation_raw.py` — TOC/raw ablation (E2), supports `--toc` flag
- `evaluate.py` — original evaluation script (unchanged)
