#!/usr/bin/env node
// Attribute per-model lost items (direct-right ∩ agent-TOC-wrong) into:
//   Cell A: direct prediction == model's preferred letter (preference-aligned correct)
//   Cell C: direct prediction != model's preferred letter (genuine knowledge candidate)
//
// Uses only:
//   result/oneshot/<model>/results.jsonl  -- direct predictions, per-item correctness
//   neurips_analysis/expert_handoff_neurips/02_MUST_HAVE_failure_analysis/Agent-TOC/<model>/q*/final_answer.txt
//   datasets/neurips_dataset_200_clean.jsonl  -- truth labels (cross-check)
//
// Does not touch the paper or any existing analysis files.

const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "..");
const OUT_DIR = __dirname;

const MODELS = [
  { key: "opus",    label: "Claude Opus 4.6",   oneshot: "neurips200_clean_oneshot_opus46",   handoff: "claude_opus_4.6" },
  { key: "sonnet",  label: "Claude Sonnet 4.6", oneshot: "neurips200_clean_oneshot_sonnet46", handoff: "claude_sonnet_4.6" },
  { key: "gpt54",   label: "GPT-5.4",           oneshot: "neurips200_clean_oneshot_gpt_5_4_", handoff: "gpt-5.4" },
  { key: "gpt52",   label: "GPT-5.2",           oneshot: "neurips200_clean_oneshot_gpt52",    handoff: "gpt-5.2" },
  { key: "gpt51",   label: "GPT-5.1",           oneshot: "neurips200_clean_oneshot_gpt51",    handoff: "gpt-5.1" },
];

const HANDOFF_BASE = path.join(REPO, "neurips_analysis/expert_handoff_neurips/02_MUST_HAVE_failure_analysis/Agent-TOC");
const ONESHOT_BASE = path.join(REPO, "result/oneshot");

function readJsonl(p) {
  return fs.readFileSync(p, "utf8")
    .trim()
    .split(/\n/)
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

function loadDirect(model) {
  const rows = readJsonl(path.join(ONESHOT_BASE, model.oneshot, "results.jsonl"));
  const map = new Map();
  for (const r of rows) {
    map.set(r.index, { prediction: r.prediction, truth: r.truth, is_correct: r.is_correct });
  }
  return map;
}

function listAgentWrongIds(model) {
  const dir = path.join(HANDOFF_BASE, model.handoff);
  return fs.readdirSync(dir)
    .filter((name) => /^q\d{5}_S\d+$/.test(name))
    .map((name) => {
      const m = name.match(/^q(\d{5})_S(\d+)$/);
      return { id: parseInt(m[1], 10), scenario: m[2], folder: name };
    });
}

function readAgentPrediction(model, folder) {
  const p = path.join(HANDOFF_BASE, model.handoff, folder, "final_answer.txt");
  if (!fs.existsSync(p)) return null;
  const txt = fs.readFileSync(p, "utf8").trim();
  const m = txt.match(/<<<\s*([A-D])\s*>>>/);
  return m ? m[1] : null;
}

function preferredLetter(directMap) {
  // Defined as the letter the model over-predicts the most relative to truth.
  const letters = ["A", "B", "C", "D"];
  const predCount = Object.fromEntries(letters.map((l) => [l, 0]));
  const truthCount = Object.fromEntries(letters.map((l) => [l, 0]));
  for (const r of directMap.values()) {
    if (predCount[r.prediction] !== undefined) predCount[r.prediction]++;
    if (truthCount[r.truth] !== undefined) truthCount[r.truth]++;
  }
  const overpred = letters.map((l) => ({ letter: l, diff: predCount[l] - truthCount[l] }));
  overpred.sort((a, b) => b.diff - a.diff);
  return { preferred: overpred[0].letter, predCount, truthCount, overpred };
}

function analyzeModel(model) {
  const direct = loadDirect(model);
  const { preferred, predCount, truthCount } = preferredLetter(direct);

  const agentWrongList = listAgentWrongIds(model);
  const agentWrongIds = new Set(agentWrongList.map((x) => x.id));

  // Sanity: total agent-wrong count
  const totalAgentWrong = agentWrongIds.size;

  // Lost = direct correct AND agent wrong
  // Counter-lost (gained) = direct wrong AND agent right
  // Kept = direct correct AND agent right
  // Missed = direct wrong AND agent wrong
  let kept = 0, lost = 0, gained = 0, missed = 0;
  let lostA = 0, lostC = 0;
  let cellA_total = 0, cellC_total = 0;

  const lostItems = [];

  for (const r of direct.values()) {
    const id = [...direct.entries()].find(([, v]) => v === r)?.[0];
    // Above is awkward; rewrite by iterating Map entries directly
  }

  for (const [id, r] of direct.entries()) {
    const directRight = r.is_correct;
    const agentWrong = agentWrongIds.has(id);
    if (directRight && !agentWrong) kept++;
    else if (directRight && agentWrong) lost++;
    else if (!directRight && !agentWrong) gained++;
    else missed++;

    // Cell totals over direct-correct items only
    if (directRight) {
      const aligned = (r.prediction === preferred);
      if (aligned) {
        cellA_total++;
        if (agentWrong) lostA++;
      } else {
        cellC_total++;
        if (agentWrong) lostC++;
      }
    }
  }

  // Per-cell loss rates and asymmetry
  const lossRateA = cellA_total > 0 ? lostA / cellA_total : null;
  const lossRateC = cellC_total > 0 ? lostC / cellC_total : null;
  const asymmetry = (lossRateA !== null && lossRateC !== null) ? lossRateC - lossRateA : null;

  // Cell A loss flip pattern: agent prediction on Cell A losses
  // (Cell A loss item: direct picked preferred, was correct; agent then got it wrong)
  const cellAFlipDist = {}; // letter -> count
  let cellAFlipNoAnswer = 0;
  for (const wrong of agentWrongList) {
    const r = direct.get(wrong.id);
    if (!r || !r.is_correct) continue;
    if (r.prediction !== preferred) continue;
    const agentPred = readAgentPrediction(model, wrong.folder);
    if (!agentPred) {
      cellAFlipNoAnswer++;
    } else {
      cellAFlipDist[agentPred] = (cellAFlipDist[agentPred] || 0) + 1;
    }
  }

  // Cell C loss flip pattern (symmetry check)
  const cellCFlipDist = {};
  let cellCFlipNoAnswer = 0;
  for (const wrong of agentWrongList) {
    const r = direct.get(wrong.id);
    if (!r || !r.is_correct) continue;
    if (r.prediction === preferred) continue;
    const agentPred = readAgentPrediction(model, wrong.folder);
    if (!agentPred) {
      cellCFlipNoAnswer++;
    } else {
      cellCFlipDist[agentPred] = (cellCFlipDist[agentPred] || 0) + 1;
    }
  }

  return {
    model: model.label,
    key: model.key,
    preferred,
    predCount,
    truthCount,
    totalAgentWrong,
    kept, lost, gained, missed,
    cellA_total, cellC_total,
    lostA, lostC,
    lossRateA, lossRateC, asymmetry,
    cellAFlipDist, cellAFlipNoAnswer,
    cellCFlipDist, cellCFlipNoAnswer,
  };
}

function pct(x, digits = 1) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "";
  return `${(100 * x).toFixed(digits)}%`;
}

function fmtFlip(dist, noAns) {
  const parts = [];
  for (const l of ["A", "B", "C", "D"]) {
    if (dist[l]) parts.push(`${l}=${dist[l]}`);
  }
  if (noAns) parts.push(`NA=${noAns}`);
  return parts.length ? parts.join(", ") : "—";
}

function main() {
  const results = MODELS.map(analyzeModel);

  // CSV
  const headers = [
    "model", "preferred",
    "kept", "lost", "gained", "missed",
    "cellA_total", "cellC_total",
    "lostA", "lostC",
    "lossRateA_pct", "lossRateC_pct", "asymmetry_pp",
    "cellA_flip", "cellC_flip",
  ];
  const lines = [headers.join(",")];
  for (const r of results) {
    lines.push([
      r.model, r.preferred,
      r.kept, r.lost, r.gained, r.missed,
      r.cellA_total, r.cellC_total,
      r.lostA, r.lostC,
      r.lossRateA !== null ? (100 * r.lossRateA).toFixed(2) : "",
      r.lossRateC !== null ? (100 * r.lossRateC).toFixed(2) : "",
      r.asymmetry !== null ? (100 * r.asymmetry).toFixed(2) : "",
      `"${fmtFlip(r.cellAFlipDist, r.cellAFlipNoAnswer)}"`,
      `"${fmtFlip(r.cellCFlipDist, r.cellCFlipNoAnswer)}"`,
    ].join(","));
  }
  fs.writeFileSync(path.join(OUT_DIR, "retention_loss_attribution.csv"), lines.join("\n") + "\n");

  // Pretty-printed JSON for full detail
  fs.writeFileSync(path.join(OUT_DIR, "retention_loss_attribution.json"), JSON.stringify(results, null, 2));

  // Console summary table
  const colW = { model: 18, pref: 4, lost: 4, A: 14, C: 14, asy: 8, flipA: 22 };
  const pad = (s, w) => String(s).padEnd(w);
  console.log(pad("Model", colW.model) + pad("Pref", colW.pref) + pad("Lost", colW.lost) +
              pad("A: lost/n (rate)", colW.A) + pad("C: lost/n (rate)", colW.C) +
              pad("C-A pp", colW.asy) + pad("A flip dist", colW.flipA));
  console.log("-".repeat(colW.model + colW.pref + colW.lost + colW.A + colW.C + colW.asy + colW.flipA));
  for (const r of results) {
    const aStr = `${r.lostA}/${r.cellA_total} (${pct(r.lossRateA)})`;
    const cStr = `${r.lostC}/${r.cellC_total} (${pct(r.lossRateC)})`;
    const asy = r.asymmetry !== null ? (100 * r.asymmetry).toFixed(1) : "—";
    console.log(pad(r.model, colW.model) + pad(r.preferred, colW.pref) + pad(r.lost, colW.lost) +
                pad(aStr, colW.A) + pad(cStr, colW.C) + pad(asy, colW.asy) +
                pad(fmtFlip(r.cellAFlipDist, r.cellAFlipNoAnswer), colW.flipA));
  }

  console.log();
  console.log("Per-model preferred-letter detail:");
  for (const r of results) {
    const op = ["A", "B", "C", "D"].map(l => `${l}=${r.predCount[l]} vs truth ${r.truthCount[l]} (Δ${r.predCount[l] - r.truthCount[l]})`).join("; ");
    console.log(`  ${r.model} preferred=${r.preferred} | ${op}`);
  }

  console.log();
  console.log("Files written:");
  console.log("  retention_loss_attribution.csv");
  console.log("  retention_loss_attribution.json");
}

main();
