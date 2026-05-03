#!/usr/bin/env node
/*
 * Reproducible analysis for the NeurIPS PHREEQC-MCQ-200 draft.
 *
 * Inputs:
 *   ../../datasets/neurips_dataset_200_clean.jsonl
 *   result directories containing results.jsonl and summary.json
 *   result/custom run workspaces containing logs/chat_*.jsonl
 *
 * Outputs are written next to this script:
 *   analysis_summary.md
 *   completed_runs.csv
 *   retention.csv
 *   scenario_metrics.csv
 *   scenario_map.csv
 *   per_option.csv
 *   toc_raw.csv
 *   model_context_windows.csv
 *   raw_cap_comparison.csv
 *   large_output_access.csv
 *   wall_clock.csv
 *   trajectory_failure.csv
 *   manual_failure_review_queue.csv
 */

const fs = require("fs");
const path = require("path");

const ANALYSIS_DIR = __dirname;
const REPO_ROOT = path.resolve(ANALYSIS_DIR, "..");
const DATASET_PATH = path.join(REPO_ROOT, "datasets", "neurips_dataset_200_clean.jsonl");

// ---------------------------------------------------------------------------
// SAFETY GUARD
// ---------------------------------------------------------------------------
// The CSVs and analysis_summary.md in this folder were generated from the
// full original benchmark run. After that run completed, per-question
// workspace artifacts for several Anthropic and original GPT TOC/Raw runs
// were lost; only failure cases remain (in expert_handoff_neurips/).
// Re-running this script with the current result/custom contents will
// produce a *reduced* subset reflecting only surviving runs and will
// overwrite the authoritative outputs cited in the paper.
//
// To prevent accidental overwrite, the script refuses to run unless the
// caller passes "--force-rerun" or the environment variable
// ALLOW_ANALYZE_RERUN is set to "1". If you need to re-run the analysis
// after re-executing the original benchmark, pass --force-rerun.
// ---------------------------------------------------------------------------
const FORCE_FLAG = process.argv.includes("--force-rerun");
const ENV_ALLOW = process.env.ALLOW_ANALYZE_RERUN === "1";
if (!FORCE_FLAG && !ENV_ALLOW) {
  console.error("ERROR: analyze_results.js refused to run.");
  console.error("");
  console.error("Running this script will overwrite the authoritative CSVs");
  console.error("and analysis_summary.md in this folder. Per-question workspace");
  console.error("artifacts for several original benchmark runs are not retained;");
  console.error("re-running here will produce a reduced subset that does not");
  console.error("match the numbers cited in the paper.");
  console.error("");
  console.error("If you have re-executed the original benchmark and intend to");
  console.error("regenerate the CSVs, pass --force-rerun:");
  console.error("    node analyze_results.js --force-rerun");
  console.error("Or set ALLOW_ANALYZE_RERUN=1 in the environment.");
  console.error("");
  console.error("See README.md for the full situation and recovery procedure.");
  process.exit(1);
}

// User-provided API context windows, recorded so raw-cap experiments can be
// interpreted as budgeted output access rather than model-independent full-output.
const MODEL_CONTEXT_WINDOWS = {
  "GPT-5.4": { context_window_tokens: 1000000, source: "user_provided_2026-05-02" },
  "GPT-5.2": { context_window_tokens: 400000, source: "user_provided_2026-05-02" },
  "GPT-5.1": { context_window_tokens: 400000, source: "user_provided_2026-05-02" },
};

const RUNS = [
  { key: "opus_direct", model: "Claude Opus 4.6", method: "Direct", family: "no_tool", rel: "result/oneshot/neurips200_clean_oneshot_opus46" },
  { key: "opus_cot", model: "Claude Opus 4.6", method: "CoT", family: "no_tool", rel: "result/oneshot/neurips200_clean_cot_opus46" },
  { key: "opus_toc", model: "Claude Opus 4.6", method: "Agent-TOC", family: "agent_toc", rel: "result/custom/neurips200_clean_toc_opus46" },

  { key: "sonnet_direct", model: "Claude Sonnet 4.6", method: "Direct", family: "no_tool", rel: "result/oneshot/neurips200_clean_oneshot_sonnet46" },
  { key: "sonnet_cot", model: "Claude Sonnet 4.6", method: "CoT", family: "no_tool", rel: "result/oneshot/neurips200_clean_cot_sonnet46" },
  { key: "sonnet_toc", model: "Claude Sonnet 4.6", method: "Agent-TOC", family: "agent_toc", rel: "result/custom/neurips200_clean_toc_sonnet46" },
  { key: "sonnet_raw", model: "Claude Sonnet 4.6", method: "Agent-Raw100k", family: "agent_raw", rel: "result/custom/neurips200_clean_agent100k_raw100k_sonnet46" },

  { key: "gpt54_direct", model: "GPT-5.4", method: "Direct", family: "no_tool", rel: "result/oneshot/neurips200_clean_oneshot_gpt_5_4_" },
  { key: "gpt54_cot", model: "GPT-5.4", method: "CoT", family: "no_tool", rel: "result/oneshot/neurips200_clean_cot_gpt_5_4_" },
  { key: "gpt54_toc", model: "GPT-5.4", method: "Agent-TOC", family: "agent_toc", rel: "result/custom/neurips200_clean_agent100k_toc_gpt_5_4_" },
  { key: "gpt54_raw", model: "GPT-5.4", method: "Agent-Raw100k", family: "agent_raw", rel: "result/custom/neurips200_clean_agent100k_raw100k_gpt_5_4_" },
  {
    key: "gpt54_raw250k",
    model: "GPT-5.4",
    method: "Agent-Raw250k",
    family: "agent_raw",
    rels: [
      "result/custom/neurips200_clean_agent_raw250k_gpt_5_4_",
      "result/custom/neurips200_clean_agent100k_raw250k_gpt_5_4_",
    ],
  },
  {
    key: "gpt54_raw500k",
    model: "GPT-5.4",
    method: "Agent-Raw500k",
    family: "agent_raw",
    rels: [
      "result/custom/neurips200_clean_agent_raw500k_gpt_5_4_",
      "result/custom/neurips200_clean_agent100k_raw500k_gpt_5_4_",
    ],
  },

  { key: "gpt52_direct", model: "GPT-5.2", method: "Direct", family: "no_tool", rel: "result/oneshot/neurips200_clean_oneshot_gpt52" },
  { key: "gpt52_cot", model: "GPT-5.2", method: "CoT", family: "no_tool", rel: "result/oneshot/neurips200_clean_cot_gpt52" },
  { key: "gpt52_toc", model: "GPT-5.2", method: "Agent-TOC", family: "agent_toc", rel: "result/custom/neurips200_clean_agent100k_toc_gpt52" },
  { key: "gpt52_raw", model: "GPT-5.2", method: "Agent-Raw100k", family: "agent_raw", rel: "result/custom/neurips200_clean_agent100k_raw100k_gpt52" },
  {
    key: "gpt52_raw500k",
    model: "GPT-5.2",
    method: "Agent-Raw500k",
    family: "agent_raw",
    rels: [
      "result/custom/neurips200_clean_agent_raw500k_gpt52",
      "result/custom/neurips200_clean_agent100k_raw500k_gpt52",
    ],
  },

  { key: "gpt51_direct", model: "GPT-5.1", method: "Direct", family: "no_tool", rel: "result/oneshot/neurips200_clean_oneshot_gpt51" },
  { key: "gpt51_cot", model: "GPT-5.1", method: "CoT", family: "no_tool", rel: "result/oneshot/neurips200_clean_cot_gpt51" },
  { key: "gpt51_toc", model: "GPT-5.1", method: "Agent-TOC", family: "agent_toc", rel: "result/custom/neurips200_clean_toc_gpt51" },
  { key: "gpt51_raw", model: "GPT-5.1", method: "Agent-Raw100k", family: "agent_raw", rel: "result/custom/neurips200_clean_agent100k_raw100k_gpt51" },
  {
    key: "gpt51_raw500k",
    model: "GPT-5.1",
    method: "Agent-Raw500k",
    family: "agent_raw",
    rels: [
      "result/custom/neurips200_clean_agent_raw500k_gpt51",
      "result/custom/neurips200_clean_agent100k_raw500k_gpt51",
    ],
  },

  { key: "gflash_direct", model: "Gemini 2.5 Flash", method: "Direct", family: "no_tool", rel: "result/oneshot/neurips200_clean_oneshot_gflash" },
  { key: "gflash_cot", model: "Gemini 2.5 Flash", method: "CoT", family: "no_tool", rel: "result/oneshot/neurips200_clean_cot_gflash" },
  { key: "gflash_toc", model: "Gemini 2.5 Flash", method: "Agent-TOC", family: "agent_toc", rel: "result/custom/neurips200_clean_agent100k_toc_gflash" },
  { key: "gflash_raw", model: "Gemini 2.5 Flash", method: "Agent-Raw100k", family: "agent_raw", rel: "result/custom/neurips200_clean_agent100k_raw100k_gflash" },

  { key: "gflash3_direct", model: "Gemini 3 Flash Preview", method: "Direct", family: "no_tool", rel: "result/oneshot/neurips200_clean_oneshot_gflash3" },
  { key: "gflash3_cot", model: "Gemini 3 Flash Preview", method: "CoT", family: "no_tool", rel: "result/oneshot/neurips200_clean_cot_gflash3" },
  { key: "gflash3_toc", model: "Gemini 3 Flash Preview", method: "Agent-TOC", family: "agent_toc", rel: "result/custom/neurips200_clean_agent100k_toc_gflash3" },
  { key: "gflash3_raw", model: "Gemini 3 Flash Preview", method: "Agent-Raw100k", family: "agent_raw", rel: "result/custom/neurips200_clean_agent100k_raw100k_gflash3" },
];

function readJsonl(file) {
  return fs.readFileSync(file, "utf8")
    .trim()
    .split(/\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function write(file, text) {
  fs.writeFileSync(path.join(ANALYSIS_DIR, file), text, "utf8");
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function toCsv(rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((c) => csvEscape(row[c])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function pct(x, digits = 1) {
  if (!Number.isFinite(x)) return "";
  return `${(100 * x).toFixed(digits)}%`;
}

function fixed(x, digits = 3) {
  if (!Number.isFinite(x)) return "";
  return x.toFixed(digits);
}

function contextWindowTokens(model) {
  return MODEL_CONTEXT_WINDOWS[model]?.context_window_tokens || "";
}

function contextWindowSource(model) {
  return MODEL_CONTEXT_WINDOWS[model]?.source || "";
}

function rawCapChars(run) {
  if (run.family !== "agent_raw") return "";
  const summaryCap = Number(run.summary.max_full_chars);
  if (Number.isFinite(summaryCap) && summaryCap > 0) return summaryCap;
  const m = run.method.match(/Raw(\d+)k/i);
  return m ? Number(m[1]) * 1000 : "";
}

function approxTokensFromChars(chars) {
  return Number.isFinite(chars) && chars > 0 ? Math.ceil(chars / 4) : "";
}

function approxCapPctOfContext(chars, model) {
  const approxTokens = approxTokensFromChars(chars);
  const windowTokens = contextWindowTokens(model);
  if (!approxTokens || !windowTokens) return "";
  return pct(approxTokens / windowTokens);
}

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function median(xs) {
  const ys = xs.slice().sort((a, b) => a - b);
  return ys[Math.floor(ys.length / 2)];
}

function mulberry32(seed) {
  return function rand() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function quantile(xs, q) {
  const ys = xs.slice().sort((a, b) => a - b);
  const pos = (ys.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return ys[lo];
  return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo);
}

function parseTimestampSeconds(ts) {
  if (!ts) return null;
  const ms = Date.parse(ts);
  return Number.isFinite(ms) ? ms / 1000 : null;
}

function scenarioStem(question) {
  const q = question.replace(/\r/g, "\n").trim();
  const paragraphs = q.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  const first = paragraphs[0] || q;
  return first.replace(/\s+/g, " ");
}

function loadRuns() {
  const loaded = [];
  for (const run of RUNS) {
    const rels = run.rels || [run.rel];
    let best = null;
    for (const rel of rels) {
      const dir = path.join(REPO_ROOT, rel);
      const resultPath = path.join(dir, "results.jsonl");
      const summaryPath = path.join(dir, "summary.json");
      if (!fs.existsSync(resultPath)) continue;
      const results = readJsonl(resultPath);
      const summary = fs.existsSync(summaryPath) ? JSON.parse(fs.readFileSync(summaryPath, "utf8")) : {};
      const candidate = { rel, dir, resultPath, summaryPath, results, summary };
      if (!best || results.length > best.results.length) best = candidate;
    }
    if (best) loaded.push({ ...run, rel: best.rel, ...best });
  }
  return loaded;
}

function getLogPath(runDir, index) {
  const qdir = path.join(runDir, "work_space", `dataset_q_${String(index).padStart(5, "0")}`, "logs");
  if (!fs.existsSync(qdir)) return null;
  const files = fs.readdirSync(qdir).filter((f) => f.endsWith(".jsonl")).sort();
  return files.length ? path.join(qdir, files[files.length - 1]) : null;
}

function getWorkspaceDir(runDir, index) {
  return path.join(runDir, "work_space", `dataset_q_${String(index).padStart(5, "0")}`);
}

function listWorkspaceFiles(runDir, index, extension) {
  const dir = getWorkspaceDir(runDir, index);
  if (!fs.existsSync(dir)) return "";
  return fs.readdirSync(dir)
    .filter((f) => f.endsWith(extension))
    .sort()
    .map((f) => path.relative(REPO_ROOT, path.join(dir, f)))
    .join(";");
}

function logStats(logPath) {
  const s = {
    steps: 0,
    exec: 0,
    execOk: 0,
    execFail: 0,
    read: 0,
    writeFinal: 0,
    runEndStatus: "missing",
    runEndStep: null,
    runStartSeconds: null,
    runEndSeconds: null,
    durationSeconds: null,
    apiError: 0,
    toolErrors: 0,
  };
  if (!logPath || !fs.existsSync(logPath)) return s;
  const lines = fs.readFileSync(logPath, "utf8").trim().split(/\n/).filter(Boolean);
  for (const line of lines) {
    let e;
    try { e = JSON.parse(line); } catch { continue; }
    if (e.action === "run_start") {
      const start = parseTimestampSeconds(e.ts);
      if (start !== null) s.runStartSeconds = start;
    }
    if (e.action === "assistant") s.steps = Math.max(s.steps, e.step || 0);
    if (e.action === "tool") {
      if (e.result && e.result.ok === false) s.toolErrors += 1;
      if (e.tool === "execute_phreeqc") {
        s.exec += 1;
        if (e.result && e.result.returncode === 0) s.execOk += 1;
        else s.execFail += 1;
      }
      if (e.tool === "read_file") s.read += 1;
      if (e.tool === "write_file" && e.args && e.args.path === "final_answer.txt") s.writeFinal += 1;
    }
    if (e.action === "run_end") {
      s.runEndStatus = e.status || "missing";
      s.runEndStep = e.step || null;
      const end = parseTimestampSeconds(e.ts);
      if (end !== null) s.runEndSeconds = end;
      if (String(e.status || "").includes("api_error")) s.apiError += 1;
    }
    if (String(e.error || "").toLowerCase().includes("api")) s.apiError += 1;
  }
  if (s.runStartSeconds !== null && s.runEndSeconds !== null && s.runEndSeconds >= s.runStartSeconds) {
    s.durationSeconds = s.runEndSeconds - s.runStartSeconds;
  }
  return s;
}

function tokenStats(logPath) {
  const s = { input_tokens: 0, output_tokens: 0 };
  if (!logPath || !fs.existsSync(logPath)) return s;
  const lines = fs.readFileSync(logPath, "utf8").trim().split(/\n/).filter(Boolean);
  for (const line of lines) {
    let e;
    try { e = JSON.parse(line); } catch { continue; }
    const usage = e.usage;
    if (!usage || typeof usage !== "object") continue;
    if (e.action === "assistant" || e.action === "aux_llm_usage") {
      s.input_tokens += usage.input_tokens || 0;
      s.output_tokens += usage.output_tokens || 0;
    }
  }
  return s;
}

function runIntegrity(run, expectedTotal) {
  if (run.results.length !== expectedTotal) {
    return {
      integrity_status: "invalid_incomplete_results",
      missing_run_end: 0,
      missing_run_end_ranges: "",
      missing_run_end_no_final: 0,
      missing_run_end_no_final_ranges: "",
    };
  }
  if (!run.family.startsWith("agent")) {
    return {
      integrity_status: "ok",
      missing_run_end: 0,
      missing_run_end_ranges: "",
      missing_run_end_no_final: 0,
      missing_run_end_no_final_ranges: "",
    };
  }
  const missing = [];
  const missingNoFinal = [];
  for (const result of run.results) {
    const st = logStats(getLogPath(run.dir, result.index));
    if (st.runEndStatus === "missing") {
      missing.push(result.index);
      const finalPath = path.join(getWorkspaceDir(run.dir, result.index), "final_answer.txt");
      if (!fs.existsSync(finalPath)) missingNoFinal.push(result.index);
    }
  }
  const invalidThreshold = Math.max(5, Math.ceil(run.results.length * 0.05));
  const status = missingNoFinal.length > invalidThreshold
    ? "invalid_missing_run_end_no_final"
    : (missingNoFinal.length ? "warning_missing_run_end_no_final" : "ok");
  return {
    integrity_status: status,
    missing_run_end: missing.length,
    missing_run_end_ranges: ranges(missing),
    missing_run_end_no_final: missingNoFinal.length,
    missing_run_end_no_final_ranges: ranges(missingNoFinal),
  };
}

function ranges(values) {
  const xs = values.slice().sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < xs.length; i++) {
    let start = xs[i];
    let end = start;
    while (i + 1 < xs.length && xs[i + 1] === end + 1) {
      i += 1;
      end = xs[i];
    }
    out.push(start === end ? String(start) : `${start}-${end}`);
  }
  return out.join(" ");
}

function classifyFailure(row, st) {
  if (row.is_correct) return "correct";
  if (!row.prediction) {
    if (st.runEndStep >= 24 || st.steps >= 24) return "ungraded_max_step_or_no_final";
    if (st.apiError) return "ungraded_api_error";
    return "ungraded_no_parse";
  }
  if (st.exec === 0) return "wrong_no_phreeqc_run";
  if (st.execOk === 0) return "wrong_all_phreeqc_failed";
  if (st.execFail > 0) return "wrong_some_phreeqc_failed";
  if (st.runEndStep >= 24 || st.steps >= 24) return "wrong_max_step_with_answer";
  return "wrong_successful_run_needs_manual_split";
}

const dataset = readJsonl(DATASET_PATH).map((row, i) => ({ ...row, index: i + 1 }));
const scenarioIds = new Map();
const scenarios = [];
for (const row of dataset) {
  const stem = scenarioStem(row.question);
  if (!scenarioIds.has(stem)) {
    const id = `S${String(scenarios.length + 1).padStart(2, "0")}`;
    scenarioIds.set(stem, id);
    scenarios.push({ scenario_id: id, stem, first_index: row.index, item_count: 0 });
  }
  const id = scenarioIds.get(stem);
  row.scenario_id = id;
  scenarios.find((s) => s.scenario_id === id).item_count += 1;
}

const runs = loadRuns();
for (const run of runs) {
  run.integrity = runIntegrity(run, dataset.length);
}
const metricRuns = runs.filter((r) => !r.integrity.integrity_status.startsWith("invalid"));
const runByKey = Object.fromEntries(metricRuns.map((r) => [r.key, r]));

// Model context windows used to interpret raw-output cap experiments.
const modelContextRows = [...new Set(RUNS.map((r) => r.model))].sort().map((model) => ({
  model,
  context_window_tokens: contextWindowTokens(model),
  source: contextWindowSource(model),
}));
write("model_context_windows.csv", toCsv(modelContextRows, [
  "model", "context_window_tokens", "source",
]));

// Completed runs table.
const completedRows = runs.map((r) => {
  const total = r.results.length;
  const correct = r.results.filter((x) => x.is_correct).length;
  const graded = r.results.filter((x) => x.prediction !== null && x.prediction !== undefined && x.prediction !== "").length;
  const capChars = rawCapChars(r);
  return {
    key: r.key,
    model: r.model,
    method: r.method,
    family: r.family,
    model_context_window_tokens: contextWindowTokens(r.model),
    raw_cap_chars: capChars,
    raw_cap_approx_tokens_chars_div_4: approxTokensFromChars(capChars),
    raw_cap_approx_pct_context_chars_div_4: approxCapPctOfContext(capChars, r.model),
    correct,
    total,
    accuracy: pct(correct / total),
    graded,
    ungraded: total - graded,
    input_tokens: r.summary.total_input_tokens || "",
    output_tokens: r.summary.total_output_tokens || "",
    elapsed_seconds: r.summary.elapsed_seconds || "",
    seconds_per_row: r.summary.seconds_per_row || "",
    integrity_status: r.integrity.integrity_status,
    missing_run_end: r.integrity.missing_run_end,
    missing_run_end_ranges: r.integrity.missing_run_end_ranges,
    missing_run_end_no_final: r.integrity.missing_run_end_no_final,
    missing_run_end_no_final_ranges: r.integrity.missing_run_end_no_final_ranges,
    result_path: path.relative(REPO_ROOT, r.resultPath),
    summary_path: fs.existsSync(r.summaryPath) ? path.relative(REPO_ROOT, r.summaryPath) : "",
  };
});
write("completed_runs.csv", toCsv(completedRows, [
  "key", "model", "method", "family",
  "model_context_window_tokens", "raw_cap_chars",
  "raw_cap_approx_tokens_chars_div_4", "raw_cap_approx_pct_context_chars_div_4",
  "correct", "total", "accuracy",
  "graded", "ungraded", "input_tokens", "output_tokens", "elapsed_seconds", "seconds_per_row",
  "integrity_status", "missing_run_end", "missing_run_end_ranges",
  "missing_run_end_no_final", "missing_run_end_no_final_ranges",
  "result_path", "summary_path",
]));

// Wall-clock timing. Agent rows use per-item run_start/run_end timestamps from
// trajectory logs. No-tool rows only have aggregate summary timing.
const wallClockRows = [];
for (const r of metricRuns) {
  const durations = [];
  if (r.family.startsWith("agent")) {
    for (const result of r.results) {
      const st = logStats(getLogPath(r.dir, result.index));
      if (Number.isFinite(st.durationSeconds)) durations.push(st.durationSeconds);
    }
  }
  wallClockRows.push({
    key: r.key,
    model: r.model,
    method: r.method,
    family: r.family,
    total_items: r.results.length,
    items_with_log_duration: durations.length,
    median_wall_seconds: durations.length ? fixed(median(durations), 1) : "",
    p90_wall_seconds: durations.length ? fixed(quantile(durations, 0.90), 1) : "",
    mean_wall_seconds: durations.length ? fixed(mean(durations), 1) : "",
    summary_elapsed_seconds: r.summary.elapsed_seconds || "",
    summary_seconds_per_row: r.summary.seconds_per_row || "",
  });
}
write("wall_clock.csv", toCsv(wallClockRows, [
  "key", "model", "method", "family", "total_items", "items_with_log_duration",
  "median_wall_seconds", "p90_wall_seconds", "mean_wall_seconds",
  "summary_elapsed_seconds", "summary_seconds_per_row",
]));

// Scenario map.
write("scenario_map.csv", toCsv(scenarios.map((s) => ({
  scenario_id: s.scenario_id,
  first_index: s.first_index,
  item_count: s.item_count,
  stem: s.stem,
})), ["scenario_id", "first_index", "item_count", "stem"]));

// Simulator output-size distribution and raw-cap coverage.
// Pick one available result.out per item from completed/loaded agent runs.
const outputByIndex = new Map();
for (const run of runs.filter((r) => r.family.startsWith("agent"))) {
  for (const result of run.results) {
    if (outputByIndex.has(result.index)) continue;
    const outPath = path.join(getWorkspaceDir(run.dir, result.index), "result.out");
    if (fs.existsSync(outPath)) {
      const size = fs.statSync(outPath).size;
      if (size > 0) outputByIndex.set(result.index, { path: outPath, size });
    }
  }
}
const outputSizes = [...outputByIndex.values()].map((x) => x.size);
if (outputSizes.length) {
  write("output_size_summary.csv", toCsv([{
    outputs: outputSizes.length,
    min_chars: Math.min(...outputSizes),
    p50_chars: Math.round(quantile(outputSizes, 0.50)),
    p75_chars: Math.round(quantile(outputSizes, 0.75)),
    p80_chars: Math.round(quantile(outputSizes, 0.80)),
    p85_chars: Math.round(quantile(outputSizes, 0.85)),
    p90_chars: Math.round(quantile(outputSizes, 0.90)),
    p95_chars: Math.round(quantile(outputSizes, 0.95)),
    p99_chars: Math.round(quantile(outputSizes, 0.99)),
    max_chars: Math.max(...outputSizes),
  }], [
    "outputs", "min_chars", "p50_chars", "p75_chars", "p80_chars",
    "p85_chars", "p90_chars", "p95_chars", "p99_chars", "max_chars",
  ]));

  const caps = [50000, 75000, 100000, 150000, 200000, 250000, 300000, 400000, 500000, 750000, 1000000];
  const charsAt100k = outputSizes.reduce((sum, size) => sum + Math.min(size, 100000), 0);
  const outputRows = [...outputByIndex.entries()].map(([index, out]) => ({ index, ...out }));
  const smallOutputs = outputRows.filter((x) => x.size <= 100000).map((x) => x.size);
  if (smallOutputs.length) {
    write("small_output_size_summary.csv", toCsv([{
      outputs_le_100k: smallOutputs.length,
      pct_outputs_le_100k: pct(smallOutputs.length / outputRows.length),
      min_chars: Math.min(...smallOutputs),
      p25_chars: Math.round(quantile(smallOutputs, 0.25)),
      p50_chars: Math.round(quantile(smallOutputs, 0.50)),
      p75_chars: Math.round(quantile(smallOutputs, 0.75)),
      p90_chars: Math.round(quantile(smallOutputs, 0.90)),
      p95_chars: Math.round(quantile(smallOutputs, 0.95)),
      max_chars: Math.max(...smallOutputs),
    }], [
      "outputs_le_100k", "pct_outputs_le_100k", "min_chars", "p25_chars",
      "p50_chars", "p75_chars", "p90_chars", "p95_chars", "max_chars",
    ]));
  }

  const sizeBins = [
    { bin: "<=10k", lo: 0, hi: 10000 },
    { bin: "10-20k", lo: 10001, hi: 20000 },
    { bin: "20-50k", lo: 20001, hi: 50000 },
    { bin: "50-100k", lo: 50001, hi: 100000 },
    { bin: ">100k", lo: 100001, hi: Infinity },
  ];
  write("output_size_bins.csv", toCsv(sizeBins.map((b) => {
    const xs = outputRows.filter((x) => x.size >= b.lo && x.size <= b.hi).map((x) => x.size);
    return {
      bin: b.bin,
      count: xs.length,
      pct_outputs: pct(xs.length / outputRows.length),
      min_chars: xs.length ? Math.min(...xs) : "",
      p50_chars: xs.length ? Math.round(quantile(xs, 0.50)) : "",
      p90_chars: xs.length ? Math.round(quantile(xs, 0.90)) : "",
      max_chars: xs.length ? Math.max(...xs) : "",
    };
  }), [
    "bin", "count", "pct_outputs", "min_chars", "p50_chars", "p90_chars", "max_chars",
  ]));

  write("output_size_caps.csv", toCsv(caps.map((cap) => {
    const exceed = outputSizes.filter((size) => size > cap).length;
    const cappedChars = outputSizes.reduce((sum, size) => sum + Math.min(size, cap), 0);
    return {
      cap_chars: cap,
      outputs_exceeding_cap: exceed,
      pct_exceeding_cap: pct(exceed / outputSizes.length),
      relative_chars_vs_100k: fixed(cappedChars / charsAt100k, 2),
    };
  }), [
    "cap_chars", "outputs_exceeding_cap", "pct_exceeding_cap", "relative_chars_vs_100k",
  ]));

  const datasetByIndexForCaps = new Map(dataset.map((r) => [r.index, r]));
  const numericPattern = /[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi;
  function answerNeedles(item) {
    const answer = item.answer || item.gold || "";
    const match = item.question.match(new RegExp(`${answer}\\.\\s*([^\\r\\n]+)`));
    const optionText = match ? match[1] : "";
    return [...new Set((optionText.match(numericPattern) || []).filter((x) => x.length >= 2))];
  }
  function answerValueState(text, item, cap) {
    const needles = answerNeedles(item);
    if (!needles.length) return "not_evaluated";
    const halfCap = Math.floor(cap / 2);
    let found = false;
    let visible = false;
    let hidden = false;
    for (const needle of needles) {
      let pos = text.indexOf(needle);
      while (pos >= 0) {
        found = true;
        if (text.length <= cap || pos < halfCap || pos >= text.length - halfCap) {
          visible = true;
        } else {
          hidden = true;
        }
        pos = text.indexOf(needle, pos + needle.length);
      }
    }
    if (!found) return "not_found";
    if (visible) return "visible";
    if (hidden) return "hidden_middle";
    return "not_found";
  }
  write("answer_value_cap_coverage.csv", toCsv(caps.filter((cap) => cap >= 100000 && cap <= 500000).map((cap) => {
    let evaluated = 0;
    let answerValueNotFound = 0;
    let answerValueInHiddenMiddle = 0;
    for (const [idx, out] of outputByIndex.entries()) {
      const item = datasetByIndexForCaps.get(idx);
      if (!item) continue;
      const text = fs.readFileSync(out.path, "utf8");
      const state = answerValueState(text, item, cap);
      if (state === "not_evaluated") continue;
      evaluated += 1;
      if (state === "not_found") answerValueNotFound += 1;
      if (state === "hidden_middle") answerValueInHiddenMiddle += 1;
    }
    return {
      cap_chars: cap,
      evaluated_items: evaluated,
      answer_value_not_found: answerValueNotFound,
      answer_value_in_hidden_middle: answerValueInHiddenMiddle,
    };
  }), [
    "cap_chars", "evaluated_items", "answer_value_not_found", "answer_value_in_hidden_middle",
  ]));

  const largeOutputs = outputRows.filter((x) => x.size > 100000);
  const largeCharsAt100k = largeOutputs.reduce((sum, out) => sum + Math.min(out.size, 100000), 0);
  write("large_output_cap_coverage.csv", toCsv(caps.filter((cap) => cap >= 100000).map((cap) => {
    let evaluated = 0;
    let answerValueNotFound = 0;
    let answerValueHidden = 0;
    let answerValueVisible = 0;
    for (const out of largeOutputs) {
      const item = datasetByIndexForCaps.get(out.index);
      if (!item) continue;
      const text = fs.readFileSync(out.path, "utf8");
      const state = answerValueState(text, item, cap);
      if (state === "not_evaluated") continue;
      evaluated += 1;
      if (state === "not_found") answerValueNotFound += 1;
      else if (state === "hidden_middle") answerValueHidden += 1;
      else if (state === "visible") answerValueVisible += 1;
    }
    const fullVisible = largeOutputs.filter((out) => out.size <= cap).length;
    const cappedChars = largeOutputs.reduce((sum, out) => sum + Math.min(out.size, cap), 0);
    return {
      cap_chars: cap,
      large_outputs: largeOutputs.length,
      fully_returned_large_outputs: fullVisible,
      still_truncated_large_outputs: largeOutputs.length - fullVisible,
      evaluated_answer_values: evaluated,
      answer_value_visible: answerValueVisible,
      answer_value_hidden_middle: answerValueHidden,
      answer_value_not_found: answerValueNotFound,
      relative_large_chars_vs_raw100k: largeCharsAt100k ? fixed(cappedChars / largeCharsAt100k, 2) : "",
    };
  }), [
    "cap_chars", "large_outputs", "fully_returned_large_outputs",
    "still_truncated_large_outputs", "evaluated_answer_values",
    "answer_value_visible", "answer_value_hidden_middle", "answer_value_not_found",
    "relative_large_chars_vs_raw100k",
  ]));
}

// Retention direct -> TOC.
const retentionPairs = [
  ["opus_direct", "opus_toc"],
  ["sonnet_direct", "sonnet_toc"],
  ["gpt54_direct", "gpt54_toc"],
  ["gpt52_direct", "gpt52_toc"],
  ["gpt51_direct", "gpt51_toc"],
  ["gflash_direct", "gflash_toc"],
  ["gflash3_direct", "gflash3_toc"],
];
const retentionRows = [];
for (const [baseKey, agentKey] of retentionPairs) {
  const base = runByKey[baseKey];
  const agent = runByKey[agentKey];
  if (!base || !agent) continue;
  const agentMap = new Map(agent.results.map((x) => [x.index, x]));
  let kept = 0, gained = 0, lost = 0, bothWrong = 0;
  for (const b of base.results) {
    const a = agentMap.get(b.index);
    if (!a) continue;
    if (b.is_correct && a.is_correct) kept += 1;
    else if (!b.is_correct && a.is_correct) gained += 1;
    else if (b.is_correct && !a.is_correct) lost += 1;
    else bothWrong += 1;
  }
  retentionRows.push({
    model: agent.model,
    baseline: base.method,
    agent: agent.method,
    baseline_correct: base.results.filter((x) => x.is_correct).length,
    agent_correct: agent.results.filter((x) => x.is_correct).length,
    kept,
    gained,
    lost,
    both_wrong: bothWrong,
    retention: pct(kept / (kept + lost)),
    net_gain: gained - lost,
  });
}
write("retention.csv", toCsv(retentionRows, [
  "model", "baseline", "agent", "baseline_correct", "agent_correct",
  "kept", "gained", "lost", "both_wrong", "retention", "net_gain",
]));

// Scenario metrics.
const itemScenario = new Map(dataset.map((r) => [r.index, r.scenario_id]));
function scenarioBootstrapCI(accs, seed = 20260502, n = 10000) {
  const rand = mulberry32(seed);
  const boot = [];
  for (let b = 0; b < n; b++) {
    let sum = 0;
    for (let i = 0; i < accs.length; i++) {
      sum += accs[Math.floor(rand() * accs.length)];
    }
    boot.push(sum / accs.length);
  }
  return [quantile(boot, 0.025), quantile(boot, 0.975)];
}
const scenarioMetricRows = [];
for (const r of metricRuns) {
  const slots = new Map(scenarios.map((s) => [s.scenario_id, { total: 0, correct: 0 }]));
  for (const result of r.results) {
    const sid = itemScenario.get(result.index);
    const slot = slots.get(sid);
    slot.total += 1;
    if (result.is_correct) slot.correct += 1;
  }
  const accs = [...slots.values()].map((s) => s.total ? s.correct / s.total : 0);
  const [lo, hi] = scenarioBootstrapCI(accs);
  scenarioMetricRows.push({
    key: r.key,
    model: r.model,
    method: r.method,
    item_correct: r.results.filter((x) => x.is_correct).length,
    item_total: r.results.length,
    item_accuracy: pct(r.results.filter((x) => x.is_correct).length / r.results.length),
    scenario_count: accs.length,
    mean_scenario_accuracy: pct(mean(accs)),
    scenario_bootstrap_ci_low: pct(lo),
    scenario_bootstrap_ci_high: pct(hi),
    all_correct_scenarios: accs.filter((x) => x === 1).length,
    any_correct_scenarios: accs.filter((x) => x > 0).length,
    all_wrong_scenarios: accs.filter((x) => x === 0).length,
  });
}
write("scenario_metrics.csv", toCsv(scenarioMetricRows, [
  "key", "model", "method", "item_correct", "item_total", "item_accuracy",
  "scenario_count", "mean_scenario_accuracy", "scenario_bootstrap_ci_low",
  "scenario_bootstrap_ci_high", "all_correct_scenarios",
  "any_correct_scenarios", "all_wrong_scenarios",
]));

// Per-option behavior.
const labels = ["A", "B", "C", "D"];
const perOptionRows = [];
for (const r of metricRuns) {
  const predCounts = {};
  for (const lab of labels) predCounts[`pred_${lab}`] = 0;
  predCounts.pred_NA = 0;
  for (const result of r.results) {
    const pred = labels.includes(result.prediction) ? result.prediction : "NA";
    predCounts[`pred_${pred}`] += 1;
  }
  for (const lab of labels) {
    const subset = r.results.filter((x) => x.truth === lab);
    const correct = subset.filter((x) => x.is_correct).length;
    perOptionRows.push({
      key: r.key,
      model: r.model,
      method: r.method,
      true_label: lab,
      true_count: subset.length,
      correct,
      accuracy: pct(correct / subset.length),
      ...predCounts,
    });
  }
}
write("per_option.csv", toCsv(perOptionRows, [
  "key", "model", "method", "true_label", "true_count", "correct", "accuracy",
  "pred_A", "pred_B", "pred_C", "pred_D", "pred_NA",
]));

// TOC vs raw pairs.
const rawPairs = [
  ["sonnet_toc", "sonnet_raw"],
  ["gpt54_toc", "gpt54_raw"],
  ["gpt54_toc", "gpt54_raw250k"],
  ["gpt54_toc", "gpt54_raw500k"],
  ["gpt52_toc", "gpt52_raw"],
  ["gpt52_toc", "gpt52_raw500k"],
  ["gpt51_toc", "gpt51_raw"],
  ["gpt51_toc", "gpt51_raw500k"],
  ["gflash_toc", "gflash_raw"],
  ["gflash3_toc", "gflash3_raw"],
];
const tocRawRows = [];
for (const [tocKey, rawKey] of rawPairs) {
  const toc = runByKey[tocKey];
  const raw = runByKey[rawKey];
  if (!toc || !raw) continue;
  const tocCorrect = toc.results.filter((x) => x.is_correct).length;
  const rawCorrect = raw.results.filter((x) => x.is_correct).length;
  const tocTokens = toc.summary.total_input_tokens || 0;
  const rawTokens = raw.summary.total_input_tokens || 0;
  const capChars = rawCapChars(raw);
  tocRawRows.push({
    model: toc.model,
    raw_method: raw.method,
    model_context_window_tokens: contextWindowTokens(raw.model),
    raw_cap_chars: capChars,
    raw_cap_approx_tokens_chars_div_4: approxTokensFromChars(capChars),
    raw_cap_approx_pct_context_chars_div_4: approxCapPctOfContext(capChars, raw.model),
    toc_accuracy: pct(tocCorrect / toc.results.length),
    raw_accuracy: pct(rawCorrect / raw.results.length),
    delta_raw_minus_toc_pp: fixed(100 * (rawCorrect / raw.results.length - tocCorrect / toc.results.length), 1),
    toc_input_tokens: tocTokens,
    raw_input_tokens: rawTokens,
    toc_token_reduction_vs_raw: rawTokens ? pct((rawTokens - tocTokens) / rawTokens) : "",
    toc_ungraded: toc.results.filter((x) => !x.prediction).length,
    raw_ungraded: raw.results.filter((x) => !x.prediction).length,
  });
}
write("toc_raw.csv", toCsv(tocRawRows, [
  "model", "raw_method", "model_context_window_tokens",
  "raw_cap_chars", "raw_cap_approx_tokens_chars_div_4", "raw_cap_approx_pct_context_chars_div_4",
  "toc_accuracy", "raw_accuracy", "delta_raw_minus_toc_pp",
  "toc_input_tokens", "raw_input_tokens", "toc_token_reduction_vs_raw",
  "toc_ungraded", "raw_ungraded",
]));

// TOC-vs-raw behavior split by observed simulator-output size.
const outputAccessBins = [
  { bin: "<=100k", lo: 0, hi: 100000 },
  { bin: ">100k", lo: 100001, hi: Infinity },
  { bin: "<=10k", lo: 0, hi: 10000 },
  { bin: "10-20k", lo: 10001, hi: 20000 },
  { bin: "20-50k", lo: 20001, hi: 50000 },
  { bin: "50-100k", lo: 50001, hi: 100000 },
  { bin: "250-500k", lo: 250001, hi: 500000 },
  { bin: ">500k", lo: 500001, hi: Infinity },
];
const outputAccessRows = [];
for (const [tocKey, rawKey] of rawPairs) {
  const toc = runByKey[tocKey];
  const raw = runByKey[rawKey];
  if (!toc || !raw) continue;
  const tocMap = new Map(toc.results.map((x) => [x.index, x]));
  const rawMap = new Map(raw.results.map((x) => [x.index, x]));
  const capChars = rawCapChars(raw);
  for (const b of outputAccessBins) {
    const ids = [...outputByIndex.entries()]
      .filter(([, out]) => out.size >= b.lo && out.size <= b.hi)
      .map(([idx]) => idx)
      .filter((idx) => tocMap.has(idx) && rawMap.has(idx));
    if (!ids.length) continue;
    let tocCorrect = 0, rawCorrect = 0, bothCorrect = 0, rawGain = 0, rawLoss = 0;
    let tocInput = 0, rawInput = 0;
    let tocUngraded = 0, rawUngraded = 0;
    for (const idx of ids) {
      const t = tocMap.get(idx);
      const r = rawMap.get(idx);
      const tc = t.is_correct === true;
      const rc = r.is_correct === true;
      if (tc) tocCorrect += 1;
      if (rc) rawCorrect += 1;
      if (tc && rc) bothCorrect += 1;
      if (!tc && rc) rawGain += 1;
      if (tc && !rc) rawLoss += 1;
      if (!t.prediction) tocUngraded += 1;
      if (!r.prediction) rawUngraded += 1;
      tocInput += tokenStats(getLogPath(toc.dir, idx)).input_tokens;
      rawInput += tokenStats(getLogPath(raw.dir, idx)).input_tokens;
    }
    outputAccessRows.push({
      model: toc.model,
      raw_method: raw.method,
      model_context_window_tokens: contextWindowTokens(raw.model),
      raw_cap_chars: capChars,
      raw_cap_approx_tokens_chars_div_4: approxTokensFromChars(capChars),
      raw_cap_approx_pct_context_chars_div_4: approxCapPctOfContext(capChars, raw.model),
      output_size_bin: b.bin,
      items: ids.length,
      toc_correct: tocCorrect,
      raw_correct: rawCorrect,
      toc_accuracy: pct(tocCorrect / ids.length),
      raw_accuracy: pct(rawCorrect / ids.length),
      raw_minus_toc_pp: fixed(100 * (rawCorrect / ids.length - tocCorrect / ids.length), 1),
      both_correct: bothCorrect,
      raw_gain: rawGain,
      raw_loss: rawLoss,
      toc_ungraded: tocUngraded,
      raw_ungraded: rawUngraded,
      toc_input_tokens: tocInput,
      raw_input_tokens: rawInput,
      toc_token_reduction_vs_raw: rawInput ? pct((rawInput - tocInput) / rawInput) : "",
      toc_input_tokens_per_item: fixed(tocInput / ids.length, 0),
      raw_input_tokens_per_item: fixed(rawInput / ids.length, 0),
    });
  }
}
write("toc_raw_by_output_size.csv", toCsv(outputAccessRows, [
  "model", "raw_method", "model_context_window_tokens",
  "raw_cap_chars", "raw_cap_approx_tokens_chars_div_4", "raw_cap_approx_pct_context_chars_div_4",
  "output_size_bin", "items",
  "toc_correct", "raw_correct", "toc_accuracy", "raw_accuracy",
  "raw_minus_toc_pp", "both_correct", "raw_gain", "raw_loss",
  "toc_ungraded", "raw_ungraded", "toc_input_tokens", "raw_input_tokens",
  "toc_token_reduction_vs_raw", "toc_input_tokens_per_item", "raw_input_tokens_per_item",
]));

// Direct raw-cap comparison for the same model, e.g. Raw100k -> Raw500k.
const rawCapPairs = [
  ["gpt54_raw", "gpt54_raw500k"],
  ["gpt52_raw", "gpt52_raw500k"],
  ["gpt51_raw", "gpt51_raw500k"],
];
const rawCapComparisonRows = [];
for (const [lowKey, highKey] of rawCapPairs) {
  const low = runByKey[lowKey];
  const high = runByKey[highKey];
  if (!low || !high) continue;
  const lowMap = new Map(low.results.map((x) => [x.index, x]));
  const highMap = new Map(high.results.map((x) => [x.index, x]));
  const lowTokens = low.summary.total_input_tokens || 0;
  const highTokens = high.summary.total_input_tokens || 0;
  const lowCap = rawCapChars(low);
  const subsets = [
    { subset: "all", keep: () => true },
    { subset: `cap_unaffected_output_le_${lowCap}`, keep: (idx) => (outputByIndex.get(idx)?.size || 0) <= lowCap },
    { subset: `cap_affected_output_gt_${lowCap}`, keep: (idx) => (outputByIndex.get(idx)?.size || 0) > lowCap },
  ];
  for (const subset of subsets) {
    let paired = 0;
    let lowCorrect = 0, highCorrect = 0, bothCorrect = 0;
    let highGain = 0, highLoss = 0, samePrediction = 0, differentPrediction = 0;
    for (const [idx, lowResult] of lowMap.entries()) {
      if (!subset.keep(idx)) continue;
      const highResult = highMap.get(idx);
      if (!highResult) continue;
      paired += 1;
      const lowIsCorrect = lowResult.is_correct === true;
      const highIsCorrect = highResult.is_correct === true;
      if (lowIsCorrect) lowCorrect += 1;
      if (highIsCorrect) highCorrect += 1;
      if (lowIsCorrect && highIsCorrect) bothCorrect += 1;
      if (!lowIsCorrect && highIsCorrect) highGain += 1;
      if (lowIsCorrect && !highIsCorrect) highLoss += 1;
      if (lowResult.prediction === highResult.prediction) samePrediction += 1;
      else differentPrediction += 1;
    }
    if (!paired) continue;
    rawCapComparisonRows.push({
      model: low.model,
      low_raw_method: low.method,
      high_raw_method: high.method,
      subset: subset.subset,
      context_window_tokens: contextWindowTokens(low.model),
      low_cap_chars: lowCap,
      high_cap_chars: rawCapChars(high),
      paired_items: paired,
      low_correct: lowCorrect,
      high_correct: highCorrect,
      low_accuracy: pct(lowCorrect / paired),
      high_accuracy: pct(highCorrect / paired),
      high_minus_low_pp: fixed(100 * (highCorrect / paired - lowCorrect / paired), 1),
      both_correct: bothCorrect,
      high_gain: highGain,
      high_loss: highLoss,
      same_prediction: samePrediction,
      different_prediction: differentPrediction,
      low_input_tokens: lowTokens,
      high_input_tokens: highTokens,
      high_token_multiplier_vs_low: lowTokens ? fixed(highTokens / lowTokens, 2) : "",
    });
  }
}
write("raw_cap_comparison.csv", toCsv(rawCapComparisonRows, [
  "model", "low_raw_method", "high_raw_method", "subset", "context_window_tokens",
  "low_cap_chars", "high_cap_chars", "paired_items",
  "low_correct", "high_correct", "low_accuracy", "high_accuracy",
  "high_minus_low_pp", "both_correct", "high_gain", "high_loss",
  "same_prediction", "different_prediction",
  "low_input_tokens", "high_input_tokens", "high_token_multiplier_vs_low",
]));

// Compact table for the cap-affected large-output subset.
const largeOutputTriples = [
  { model: "GPT-5.4", toc: "gpt54_toc", raw100: "gpt54_raw", raw500: "gpt54_raw500k" },
  { model: "GPT-5.2", toc: "gpt52_toc", raw100: "gpt52_raw", raw500: "gpt52_raw500k" },
  { model: "GPT-5.1", toc: "gpt51_toc", raw100: "gpt51_raw", raw500: "gpt51_raw500k" },
];
const largeIds = [...outputByIndex.entries()]
  .filter(([, out]) => out.size > 100000)
  .map(([idx]) => idx);
function subsetRunStats(run, ids) {
  if (!run) return null;
  const resultMap = new Map(run.results.map((x) => [x.index, x]));
  const present = ids.filter((idx) => resultMap.has(idx));
  let correct = 0, ungraded = 0, tokens = 0;
  for (const idx of present) {
    const result = resultMap.get(idx);
    if (result.is_correct) correct += 1;
    if (!result.prediction) ungraded += 1;
    tokens += tokenStats(getLogPath(run.dir, idx)).input_tokens;
  }
  return {
    n: present.length,
    correct,
    accuracy: pct(correct / present.length),
    ungraded,
    input_tokens: tokens,
    input_tokens_per_item: fixed(tokens / present.length, 0),
  };
}
const largeOutputAccessRows = [];
for (const triple of largeOutputTriples) {
  const toc = runByKey[triple.toc];
  const raw100 = runByKey[triple.raw100];
  const raw500 = runByKey[triple.raw500];
  if (!toc || !raw100 || !raw500) continue;
  const tocStats = subsetRunStats(toc, largeIds);
  const raw100Stats = subsetRunStats(raw100, largeIds);
  const raw500Stats = subsetRunStats(raw500, largeIds);
  largeOutputAccessRows.push({
    model: triple.model,
    context_window_tokens: contextWindowTokens(triple.model),
    items: largeIds.length,
    toc_accuracy: tocStats.accuracy,
    raw100_accuracy: raw100Stats.accuracy,
    raw500_accuracy: raw500Stats.accuracy,
    toc_input_tokens_per_item: tocStats.input_tokens_per_item,
    raw100_input_tokens_per_item: raw100Stats.input_tokens_per_item,
    raw500_input_tokens_per_item: raw500Stats.input_tokens_per_item,
    toc_ungraded: tocStats.ungraded,
    raw100_ungraded: raw100Stats.ungraded,
    raw500_ungraded: raw500Stats.ungraded,
  });
}
write("large_output_access.csv", toCsv(largeOutputAccessRows, [
  "model", "context_window_tokens", "items",
  "toc_accuracy", "raw100_accuracy", "raw500_accuracy",
  "toc_input_tokens_per_item", "raw100_input_tokens_per_item", "raw500_input_tokens_per_item",
  "toc_ungraded", "raw100_ungraded", "raw500_ungraded",
]));

// Trajectory and automatic failure taxonomy for custom runs.
const trajRows = [];
const failureRows = [];
for (const r of metricRuns.filter((x) => x.family.startsWith("agent"))) {
  const steps = [], execs = [], reads = [], durations = [];
  let failedRunEvents = 0;
  const failureCounts = {};
  for (const result of r.results) {
    const st = logStats(getLogPath(r.dir, result.index));
    steps.push(st.steps);
    execs.push(st.exec);
    reads.push(st.read);
    if (Number.isFinite(st.durationSeconds)) durations.push(st.durationSeconds);
    failedRunEvents += st.execFail;
    const cls = classifyFailure(result, st);
    failureCounts[cls] = (failureCounts[cls] || 0) + 1;
  }
  trajRows.push({
    key: r.key,
    model: r.model,
    method: r.method,
    median_steps: median(steps),
    mean_steps: fixed(mean(steps), 2),
    median_phreeqc_runs: median(execs),
    mean_phreeqc_runs: fixed(mean(execs), 2),
    median_read_file_calls: median(reads),
    mean_read_file_calls: fixed(mean(reads), 2),
    failed_phreeqc_run_events: failedRunEvents,
    items_with_wall_clock: durations.length,
    median_wall_seconds: durations.length ? fixed(median(durations), 1) : "",
    p90_wall_seconds: durations.length ? fixed(quantile(durations, 0.90), 1) : "",
  });
  for (const [category, count] of Object.entries(failureCounts).sort()) {
    failureRows.push({
      key: r.key,
      model: r.model,
      method: r.method,
      category,
      count,
    });
  }
}
write("trajectory.csv", toCsv(trajRows, [
  "key", "model", "method", "median_steps", "mean_steps",
  "median_phreeqc_runs", "mean_phreeqc_runs",
  "median_read_file_calls", "mean_read_file_calls",
  "failed_phreeqc_run_events", "items_with_wall_clock",
  "median_wall_seconds", "p90_wall_seconds",
]));
write("trajectory_failure.csv", toCsv(failureRows, [
  "key", "model", "method", "category", "count",
]));

// Review queue for domain/manual error analysis. This is intentionally broad:
// it points reviewers to artifacts but does not assign domain root causes.
const datasetByIndex = new Map(dataset.map((r) => [r.index, r]));
const manualQueueRows = [];
for (const r of metricRuns.filter((x) => x.family.startsWith("agent"))) {
  for (const result of r.results) {
    if (result.is_correct) continue;
    const st = logStats(getLogPath(r.dir, result.index));
    const item = datasetByIndex.get(result.index);
    manualQueueRows.push({
      key: r.key,
      model: r.model,
      method: r.method,
      item_index: result.index,
      scenario_id: item ? item.scenario_id : "",
      truth: result.truth,
      prediction: result.prediction || "",
      automatic_category: classifyFailure(result, st),
      phreeqc_runs: st.exec,
      phreeqc_successes: st.execOk,
      phreeqc_failures: st.execFail,
      steps: st.steps,
      workspace_dir: path.relative(REPO_ROOT, getWorkspaceDir(r.dir, result.index)),
      log_path: getLogPath(r.dir, result.index) ? path.relative(REPO_ROOT, getLogPath(r.dir, result.index)) : "",
      pqi_files: listWorkspaceFiles(r.dir, result.index, ".pqi"),
      out_files: listWorkspaceFiles(r.dir, result.index, ".out"),
      question: item ? item.question.replace(/\s+/g, " ") : "",
    });
  }
}
write("manual_failure_review_queue.csv", toCsv(manualQueueRows, [
  "key", "model", "method", "item_index", "scenario_id", "truth", "prediction",
  "automatic_category", "phreeqc_runs", "phreeqc_successes", "phreeqc_failures",
  "steps", "workspace_dir", "log_path", "pqi_files", "out_files", "question",
]));

// Markdown summary for human traceability.
const lines = [];
lines.push("# NeurIPS Analysis Summary");
lines.push("");
lines.push(`Generated: ${new Date().toISOString()}`);
lines.push(`Dataset: \`${path.relative(REPO_ROOT, DATASET_PATH)}\``);
lines.push(`Question items: ${dataset.length}`);
lines.push(`Scenario stems: ${scenarios.length}`);
lines.push("");
lines.push("## Completed Runs");
lines.push("");
lines.push("| Model | Method | Correct | Accuracy | Ungraded | Input tokens | Integrity |");
lines.push("| --- | --- | ---: | ---: | ---: | ---: | --- |");
for (const r of completedRows) {
  const status = r.integrity_status === "ok" ? "ok" : `${r.integrity_status} (${r.missing_run_end_no_final_ranges})`;
  lines.push(`| ${r.model} | ${r.method} | ${r.correct}/${r.total} | ${r.accuracy} | ${r.ungraded} | ${r.input_tokens || ""} | ${status} |`);
}
lines.push("");
lines.push("## Wall-Clock Timing (Agent Runs)");
lines.push("");
lines.push("| Model | Method | Items timed | Median sec/item | p90 sec/item |");
lines.push("| --- | --- | ---: | ---: | ---: |");
for (const r of wallClockRows.filter((x) => x.family.startsWith("agent") && x.items_with_log_duration)) {
  lines.push(`| ${r.model} | ${r.method} | ${r.items_with_log_duration}/${r.total_items} | ${r.median_wall_seconds} | ${r.p90_wall_seconds} |`);
}
lines.push("");
lines.push("## Retention: Direct -> Agent-TOC");
lines.push("");
lines.push("| Model | Kept | Gained | Lost | Retention | Net gain |");
lines.push("| --- | ---: | ---: | ---: | ---: | ---: |");
for (const r of retentionRows) {
  lines.push(`| ${r.model} | ${r.kept} | ${r.gained} | ${r.lost} | ${r.retention} | ${r.net_gain >= 0 ? "+" : ""}${r.net_gain} |`);
}
lines.push("");
lines.push("## Scenario-Level Metrics (Selected)");
lines.push("");
lines.push("| Run | Item acc. | Mean scenario acc. | 95% CI | All-correct | All-wrong |");
lines.push("| --- | ---: | ---: | ---: | ---: | ---: |");
for (const key of ["sonnet_toc", "opus_toc", "gpt54_toc", "gpt52_toc", "opus_direct"]) {
  const r = scenarioMetricRows.find((x) => x.key === key);
  if (!r) continue;
  lines.push(`| ${r.model} ${r.method} | ${r.item_accuracy} | ${r.mean_scenario_accuracy} | [${r.scenario_bootstrap_ci_low}, ${r.scenario_bootstrap_ci_high}] | ${r.all_correct_scenarios} | ${r.all_wrong_scenarios} |`);
}
lines.push("");
lines.push("## Notes");
lines.push("");
lines.push("- Scenario stems are generated by taking the first paragraph of each question after whitespace normalization.");
lines.push("- Scenario uncertainty is a nonparametric bootstrap over scenario stems, not over individual items.");
lines.push("- Runs marked `invalid_*` fail log-integrity checks and are excluded from downstream metric tables.");
lines.push("- Automatic failure categories are log-derived and should be treated as a review queue, not as final domain-root-cause labels.");
lines.push("- `manual_failure_review_queue.csv` collects wrong or ungraded agent cases with artifact paths for later expert labeling.");
lines.push("- `wall_clock.csv` reports aggregate timing for all runs and per-item median/p90 wall-clock time for agent runs with trajectory logs.");
write("analysis_summary.md", `${lines.join("\n")}\n`);

console.log(`Wrote analysis outputs to ${ANALYSIS_DIR}`);
