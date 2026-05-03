#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.resolve(__dirname, "..");
const OUT = path.join(REPO_ROOT, "neurips_analysis", "expert_handoff_neurips");

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function csvEscape(value) {
  const s = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i], next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') inQuotes = false;
      else cell += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (ch !== "\r") cell += ch;
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((r) => r.length && !(r.length === 1 && r[0] === ""));
}

function readCsv(rel) {
  const rows = parseCsv(fs.readFileSync(path.join(REPO_ROOT, rel), "utf8"));
  const header = rows[0];
  return rows.slice(1).map((row) => Object.fromEntries(header.map((h, i) => [h, row[i] || ""])));
}

function writeCsv(rel, columns, rows) {
  const outPath = path.join(OUT, rel);
  ensureDir(path.dirname(outPath));
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((c) => csvEscape(row[c])).join(","));
  fs.writeFileSync(outPath, `${lines.join("\n")}\n`, "utf8");
}

function writeText(rel, text) {
  const outPath = path.join(OUT, rel);
  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, text, "utf8");
}

function copyFile(srcRel, dstRel) {
  const src = path.join(REPO_ROOT, srcRel);
  const dst = path.join(OUT, dstRel);
  ensureDir(path.dirname(dst));
  if (fs.existsSync(src)) fs.copyFileSync(src, dst);
}

function copyDir(srcRel, dstRel) {
  const src = path.join(REPO_ROOT, srcRel);
  const dst = path.join(OUT, dstRel);
  if (!fs.existsSync(src)) return false;
  ensureDir(path.dirname(dst));
  fs.cpSync(src, dst, { recursive: true });
  return true;
}

function safeName(s) {
  return String(s).replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
}

function readJsonl(rel) {
  return fs.readFileSync(path.join(REPO_ROOT, rel), "utf8")
    .trim()
    .split(/\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function scenarioStem(question) {
  const paragraphs = question.replace(/\r/g, "\n").trim().split(/\n\s*\n/);
  return (paragraphs[0] || question).replace(/\s+/g, " ").trim();
}

function questionTarget(question) {
  const parts = question.replace(/\r/g, "\n").trim().split(/\n\s*\n/)
    .map((s) => s.replace(/\s+/g, " ").trim())
    .filter(Boolean);
  return parts[1] || parts[0] || "";
}

function methodFolderName(method) {
  if (method === "Agent-TOC") return "Agent-TOC";
  if (method === "Agent-Raw100k") return "Agent-Raw100k";
  if (method === "Agent-Raw500k") return "Agent-Raw500k";
  return safeName(method);
}

const preservedFiles = [
  ["readme.md", ".expert_handoff_readme_backup.md"],
  ["01_MUST_HAVE_scenarios_check/scenarios_check.md", ".expert_handoff_scenarios_check_backup.md"],
  ["02_MUST_HAVE_failure_analysis/README.md", ".expert_handoff_failure_readme_backup.md"],
];
for (const [rel, backupName] of preservedFiles) {
  const src = path.join(OUT, rel);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, path.join(REPO_ROOT, "neurips_analysis", backupName));
  }
}
if (fs.existsSync(OUT)) fs.rmSync(OUT, { recursive: true, force: true });

const folders = [
  "dataset",
  "01_MUST_HAVE_scenarios_check/scenario_items",
  "01_MUST_HAVE_scenarios_check/csv_reference",
  "02_MUST_HAVE_failure_analysis/Agent-TOC",
  "02_MUST_HAVE_failure_analysis/Agent-Raw100k",
  "02_MUST_HAVE_failure_analysis/Agent-Raw500k",
  "05_OPTIONAL_question_analysis",
  "99_OPTIONAL_full_materials",
];
for (const folder of folders) ensureDir(path.join(OUT, folder));
for (const [rel, backupName] of preservedFiles) {
  const backup = path.join(REPO_ROOT, "neurips_analysis", backupName);
  if (fs.existsSync(backup)) {
    const dst = path.join(OUT, rel);
    ensureDir(path.dirname(dst));
    fs.copyFileSync(backup, dst);
    fs.rmSync(backup, { force: true });
  }
}

const dataset = readJsonl("datasets/neurips_dataset_200_clean.jsonl");
const scenarioMap = readCsv("neurips_analysis/scenario_map.csv");
const scenarioById = new Map(scenarioMap.map((s) => [s.scenario_id, s]));
const scenarioIdByStem = new Map(scenarioMap.map((s) => [s.stem, s.scenario_id]));
for (const item of dataset) {
  item.scenario_id = scenarioIdByStem.get(scenarioStem(item.question)) || "";
  item.item_id = item.id || item.index || item.original_row_index;
}

copyFile("datasets/neurips_dataset_200_clean.jsonl", "dataset/neurips_dataset_200_clean.jsonl");

if (!fs.existsSync(path.join(OUT, "readme.md"))) {
writeText("readme.md", `# Expert Handoff: PHREEQC-MCQ-200 NeurIPS

Please start with the MUST folders. Optional folders contain reference tables and full materials.

## Folder Map

- \`dataset\`: frozen 200-item dataset plus readable Markdown.
- \`01_MUST_HAVE_scenarios_check\`: confirm the automatic scenario grouping.
- \`02_MUST_HAVE_failure_analysis\`: label Claude/GPT agent error root causes, including TOC and raw-output runs.
- \`05_OPTIONAL_question_analysis\`: question-level solved/unsolved groups.
- \`99_OPTIONAL_full_materials\`: database, previous ACM source, and full queues.

Please do not feel constrained by our labels. If a category is wrong, use \`other\` and write notes.
`);
}

const readableDataset = ["# PHREEQC-MCQ-200 Readable Dataset", ""];
for (const item of dataset) {
  readableDataset.push(`## Q${item.item_id} (${item.scenario_id || "unknown"})`);
  readableDataset.push("");
  readableDataset.push(item.question.trim());
  readableDataset.push("");
  readableDataset.push(`Truth: ${item.answer || item.truth || item.correct_answer || ""}`);
  readableDataset.push("");
}
writeText("dataset/dataset_200_readable.md", `${readableDataset.join("\n")}\n`);

const scenarioFillRows = scenarioMap.map((s) => ({
  scenario_id: s.scenario_id,
  first_index: s.first_index,
  item_count: s.item_count,
  stem: s.stem,
  expert_confirm_grouping: "",
  expert_short_label: "",
  expert_broad_category: "",
  expert_notes: "",
}));
writeCsv("01_MUST_HAVE_scenarios_check/csv_reference/scenarios_check_fill.csv", [
  "scenario_id", "first_index", "item_count", "stem",
  "expert_confirm_grouping", "expert_short_label", "expert_broad_category", "expert_notes",
], scenarioFillRows);
copyFile("neurips_analysis/scenario_map.csv", "01_MUST_HAVE_scenarios_check/csv_reference/scenario_map_auto.csv");

const scenarioMd = [
  "# Scenario Check",
  "",
  "Priority: MUST HAVE",
  "",
  "Please read the scenario blocks below and confirm whether the automatic grouping is scientifically reasonable.",
  "",
  "Write short notes directly under each block. The CSV files are in `csv_reference/` only for reference.",
  "",
];
for (const s of scenarioMap) {
  const items = dataset.filter((item) => item.scenario_id === s.scenario_id);
  scenarioMd.push(`## ${s.scenario_id}: items ${s.first_index} - ${Number(s.first_index) + Number(s.item_count) - 1}`);
  scenarioMd.push("");
  scenarioMd.push(`Auto stem: ${s.stem}`);
  scenarioMd.push("");
  scenarioMd.push("Expert notes:");
  scenarioMd.push("");
  scenarioMd.push("- Confirm grouping: ");
  scenarioMd.push("- Short label: ");
  scenarioMd.push("- Broad category: ");
  scenarioMd.push("- Split/merge notes: ");
  scenarioMd.push("");
  scenarioMd.push("Items:");
  for (const item of items) scenarioMd.push(`- Q${item.item_id}: truth ${item.answer || item.truth || ""}`);
  scenarioMd.push("");

  const itemMd = [`# ${s.scenario_id}`, "", `Auto stem: ${s.stem}`, ""];
  for (const item of items) {
    itemMd.push(`## Q${item.item_id}`);
    itemMd.push("");
    itemMd.push(item.question.trim());
    itemMd.push("");
    itemMd.push(`Truth: ${item.answer || item.truth || ""}`);
    itemMd.push("");
  }
  writeText(`01_MUST_HAVE_scenarios_check/scenario_items/${s.scenario_id}.md`, `${itemMd.join("\n")}\n`);
}
if (!fs.existsSync(path.join(OUT, "01_MUST_HAVE_scenarios_check", "scenarios_check.md"))) {
  writeText("01_MUST_HAVE_scenarios_check/scenarios_check.md", `${scenarioMd.join("\n")}\n`);
}

const queue = readCsv("neurips_analysis/manual_failure_review_queue.csv");
const failureClaudeGpt = queue.filter((r) =>
  (r.model.startsWith("Claude") || r.model.startsWith("GPT-")) &&
  ["Agent-TOC", "Agent-Raw100k", "Agent-Raw500k"].includes(r.method)
);

if (!fs.existsSync(path.join(OUT, "02_MUST_HAVE_failure_analysis", "README.md"))) {
  writeText("02_MUST_HAVE_failure_analysis/README.md", `# Failure Analysis
Priority: MUST HAVE

Hi Sahchit, this is the failure analysis section, which you have done for ACM once.

The following are the labels you used for ACM, I keep it here:
- \`wrong_interpretation\`
- \`some_phreeqc_failed\`
- \`all_phreeqc_failed\`
- \`max_steps\`
- \`timeout/error\`
- \`no_phreeqc_run\`
- \`Bad PHREEQC input\`
- \`Precision limit\`
- \`PHREEQC retry loop\`
- \`complex_setup_failure\`
- \`other\`

I think we can borrow you paragraphs from ACM, but since we have new dataset now, I am now sure if we gonna have the same conclusion so better go over it.
Also please try to detect if there are any failre difference among Agent-TOC, Agent-Raw100k, and Agent-Raw500k.



1. Overall taxonomy / main failure modes:

2. Differences between TOC and raw-output failures:

3. Important PHREEQC-specific causes:

4. Simulator-general causes likely relevant beyond PHREEQC:

5. Any dataset/question problems:

6. Paper-ready paragraph if useful:

________________________________________________________________
Anything you want to say:
`);
}

function failureWorkspaceRel(row) {
  const modelFolder = safeName(row.model.toLowerCase());
  const caseFolder = `q${String(row.item_index).padStart(5, "0")}_${row.scenario_id}`;
  return `02_MUST_HAVE_failure_analysis/${methodFolderName(row.method)}/${modelFolder}/${caseFolder}`;
}

for (const row of failureClaudeGpt) copyDir(row.workspace_dir, failureWorkspaceRel(row));

function methodKey(method) {
  if (method === "Direct") return "Direct";
  if (method === "CoT") return "CoT";
  if (method === "Agent-TOC") return "Agent-TOC";
  if (method === "Agent-Raw100k") return "Agent-Raw100k";
  if (method === "Agent-Raw500k") return "Agent-Raw500k";
  return "";
}

const runsByKey = Object.fromEntries(readCsv("neurips_analysis/completed_runs.csv").map((r) => [r.key, r]));
const modelSpecs = [
  { label: "Claude Sonnet 4.6", note: "Raw500k not run; agent group uses TOC and Raw100k.", keys: { direct: "sonnet_direct", cot: "sonnet_cot", toc: "sonnet_toc", raw100k: "sonnet_raw", raw500k: null } },
  { label: "GPT-5.1", note: "Complete five-mode comparison.", keys: { direct: "gpt51_direct", cot: "gpt51_cot", toc: "gpt51_toc", raw100k: "gpt51_raw", raw500k: "gpt51_raw500k" } },
  { label: "GPT-5.2", note: "Complete five-mode comparison.", keys: { direct: "gpt52_direct", cot: "gpt52_cot", toc: "gpt52_toc", raw100k: "gpt52_raw", raw500k: "gpt52_raw500k" } },
  { label: "GPT-5.4", note: "Complete five-mode comparison.", keys: { direct: "gpt54_direct", cot: "gpt54_cot", toc: "gpt54_toc", raw100k: "gpt54_raw", raw500k: "gpt54_raw500k" } },
];
const resultByRun = new Map();
for (const spec of modelSpecs) {
  for (const key of Object.values(spec.keys).filter(Boolean)) {
    const run = runsByKey[key];
    resultByRun.set(key, new Map(readJsonl(run.result_path).map((r) => [Number(r.index), r])));
  }
}
function correct(key, index) {
  return key ? resultByRun.get(key)?.get(index)?.is_correct === true : false;
}
const questionRows = [];
for (const spec of modelSpecs) {
  for (const item of dataset) {
    const idx = Number(item.id || item.index || item.original_row_index);
    const q = item.question || "";
    const direct = correct(spec.keys.direct, idx);
    const cot = correct(spec.keys.cot, idx);
    const toc = correct(spec.keys.toc, idx);
    const raw100k = correct(spec.keys.raw100k, idx);
    const raw500k = correct(spec.keys.raw500k, idx);
    const noTool = direct || cot;
    const agent = toc || raw100k || raw500k;
    const category = noTool && agent ? "both" : (!noTool && agent ? "agent_only" : (noTool && !agent ? "direct_cot_only" : "neither"));
    questionRows.push({
      model: spec.label,
      item_index: idx,
      scenario_id: scenarioIdByStem.get(scenarioStem(q)) || "",
      truth: item.answer || item.truth || "",
      category,
      direct_correct: direct ? 1 : 0,
      cot_correct: cot ? 1 : 0,
      no_tool_solved: noTool ? 1 : 0,
      toc_correct: toc ? 1 : 0,
      raw100k_correct: raw100k ? 1 : 0,
      raw500k_available: spec.keys.raw500k ? 1 : 0,
      raw500k_correct: spec.keys.raw500k ? (raw500k ? 1 : 0) : "",
      agent_solved: agent ? 1 : 0,
      toc_only_agent: toc && !raw100k && !raw500k ? 1 : 0,
      raw_only_agent: !toc && (raw100k || raw500k) ? 1 : 0,
      raw500k_only_agent: raw500k && !toc && !raw100k ? 1 : 0,
      question_target: questionTarget(q),
    });
  }
}
writeCsv("05_OPTIONAL_question_analysis/question_analysis_by_model.csv", [
  "model", "item_index", "scenario_id", "truth", "category",
  "direct_correct", "cot_correct", "no_tool_solved",
  "toc_correct", "raw100k_correct", "raw500k_available", "raw500k_correct", "agent_solved",
  "toc_only_agent", "raw_only_agent", "raw500k_only_agent", "question_target",
], questionRows);

const categoryDefs = [
  ["both", "Both Direct/CoT and Agent solve", "Direct or CoT is correct, and at least one available agent mode is correct."],
  ["agent_only", "Agent-only solved", "No Direct/CoT run is correct, but at least one available agent mode is correct."],
  ["direct_cot_only", "Direct/CoT-only solved", "Direct or CoT is correct, but no available agent mode is correct."],
  ["neither", "Unsolved by both groups", "No Direct/CoT or available agent mode is correct for this model."],
];
function questionLabel(row) {
  const flags = [
    row.direct_correct === 1 ? "D" : "",
    row.cot_correct === 1 ? "CoT" : "",
    row.toc_correct === 1 ? "TOC" : "",
    row.raw100k_correct === 1 ? "R100" : "",
    row.raw500k_correct === 1 ? "R500" : "",
  ].filter(Boolean).join("/");
  return `Q${row.item_index} (${row.scenario_id || "S??"}, truth ${row.truth}, ${flags || "none"}): ${row.question_target}`;
}
const questionMd = [
  "# Question Analysis By Model",
  "",
  "Priority: OPTIONAL",
  "",
  "Question-level comparison by backbone. Sonnet is included with its available modes; GPT backbones have complete five-mode comparisons.",
  "",
  "Definitions:",
  "",
  "- Direct/CoT group: Direct or CoT is correct for the same model.",
  "- Agent group: any available agent mode is correct for the same model.",
  "- Flags in parentheses: D=Direct, CoT=chain-of-thought, TOC=Agent-TOC, R100=Agent-Raw100k, R500=Agent-Raw500k.",
  "",
];
for (const spec of modelSpecs) {
  const rows = questionRows.filter((r) => r.model === spec.label);
  questionMd.push(`## ${spec.label}`, "", spec.note, "", "Summary:", "");
  for (const [key, label] of categoryDefs) questionMd.push(`- ${label}: ${rows.filter((r) => r.category === key).length}`);
  questionMd.push(`- TOC-only agent solved: ${rows.filter((r) => r.toc_only_agent === 1).length}`);
  questionMd.push(`- Raw-only agent solved: ${rows.filter((r) => r.raw_only_agent === 1).length}`);
  if (spec.keys.raw500k) questionMd.push(`- Raw500k-only agent solved: ${rows.filter((r) => r.raw500k_only_agent === 1).length}`);
  questionMd.push("");
  for (const [key, label, description] of categoryDefs) {
    const groupRows = rows.filter((r) => r.category === key);
    questionMd.push(`### ${label} (${groupRows.length})`, "", description, "");
    if (groupRows.length === 0) questionMd.push("_None._", "");
    else {
      for (const row of groupRows) questionMd.push(`- ${questionLabel(row)}`);
      questionMd.push("");
    }
  }
}
writeText("05_OPTIONAL_question_analysis/question_groups_by_model.md", `${questionMd.join("\n")}\n`);
writeText("05_OPTIONAL_question_analysis/README.md", `# Question Analysis

Priority: OPTIONAL

This folder summarizes question-level behavior by backbone.

- \`question_groups_by_model.md\`: Sonnet, GPT-5.1, GPT-5.2, and GPT-5.4 sections comparing Direct/CoT vs Agent.
- \`question_analysis_by_model.csv\`: one row per model-question pair with Direct, CoT, TOC, Raw100k, and Raw500k flags where available.

No need to write here unless a group looks scientifically suspicious.
`);

copyFile("archive_acm_2026/all_materials/phreeqc.dat", "99_OPTIONAL_full_materials/phreeqc_author_supplied_database.dat");
copyFile("paper_neurips_ed/main.tex", "99_OPTIONAL_full_materials/current_main_tex.tex");
copyFile("paper_neurips_ed/results_macros.tex", "99_OPTIONAL_full_materials/current_results_macros.tex");
copyFile("archive_acm_2026/acm_summary/ACM_paper.tex", "99_OPTIONAL_full_materials/previous_acm_paper.tex");
copyFile("neurips_analysis/manual_failure_review_queue.csv", "99_OPTIONAL_full_materials/manual_failure_review_queue_all_runs.csv");

console.log(`Prepared ${OUT}`);
