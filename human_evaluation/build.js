#!/usr/bin/env node
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert/strict');
const dir = __dirname;
const repo = path.dirname(dir);
const sourcePath = path.join(repo, 'datasets/neurips_dataset_200_clean.jsonl');
const dbPath = path.join(repo, 'archive_acm_2026/all_materials/phreeqc.dat');
const sourceBytes = fs.readFileSync(sourcePath);
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const rows = sourceBytes.toString('utf8').trim().split(/\r?\n/).map(JSON.parse);
const configs = JSON.parse(fs.readFileSync(path.join(dir, 'scenarios.json'), 'utf8'));
const verificationBytes = fs.readFileSync(path.join(dir, 'verification.json'));
const verification = JSON.parse(verificationBytes);
const statusOf = new Map(verification.questions.map(q => [q.id, q.status]));
const excluded = new Set(verification.questions.filter(q => q.status !== 'verified').map(q => q.id));
const seed = 'phreeqc-human-100-v2-20260930';
const meta = {
  study_id: 'phreeqc-human-100-v2',
  pool_rule: 'phreeqc_verified_items_only',
  source_file: path.basename(sourcePath),
  source_sha256: sha(sourceBytes),
  database_sha256: sha(fs.readFileSync(dbPath)),
  verification_sha256: sha(verificationBytes),
  phreeqc_version: '3.8.6',
  seed,
  sampling: 'Eligible items are those whose answer key was reproduced by an independent PHREEQC 3.8.6 run with the frozen database (status "verified" in verification.json). Within each scenario, rank eligible items by SHA-256(seed|scenario_id|question_id), take the five smallest hashes, and display them in original ID order. Scenarios marked "hold" in scenarios.json (scenario-level text defects) are not sampled. Answer letters and model outcomes do not enter the ranking.',
  questions_per_scenario: 5,
  original_question_count: rows.length, excluded_ids: [...excluded].sort((a, b) => a - b)
};
assert.equal(rows.length, 200);
assert.equal(new Set(rows.map(r => r.id)).size, 200);
assert.equal(configs.length, 20);
const held = configs.filter(c => c.hold);
const active = configs.filter(c => !c.hold);
meta.scenario_count = active.length;
meta.question_count = active.length * meta.questions_per_scenario;
meta.held_scenarios = held.map(c => ({id: c.id, reason: c.hold}));
assert.equal(statusOf.size, 200, 'verification.json must cover all 200 questions once');
assert(rows.every(r => statusOf.has(r.id)), 'verification.json is missing a question ID');
assert.equal(verification.database_sha256, meta.database_sha256, 'Verification used a different database');
function parse(r) {
  const text = r.question.replace(/\r\n/g, '\n');
  const breakAt = text.search(/\n\s*\n/);
  assert(breakAt > 0, 'No preamble boundary for Q' + r.id);
  const preamble = text.slice(0, breakAt);
  const remainder = text.slice(breakAt).trimStart();
  const optionAt = remainder.search(/^A\.[ \t]*/m);
  assert(optionAt >= 0, 'No options for Q' + r.id);
  const target = remainder.slice(0, optionAt).trim();
  const matches = [...remainder.slice(optionAt).matchAll(/^([A-D])\.[ \t]*(.*)$/gm)];
  assert.deepEqual(matches.map(m => m[1]), ['A', 'B', 'C', 'D'], 'Bad options Q' + r.id);
  const options = Object.fromEntries(matches.map(m => [m[1], m[2]]));
  if (!target) assert(excluded.has(r.id), 'Question without a stem is marked verified: Q' + r.id);
  return {id: r.id, preamble, target, options, source_question: r.question};
}
const parsed = new Map(rows.map(r => [r.id, parse(r)]));
const seen = new Set();
const all = configs.map(c => {
  const originals = rows.filter(r => r.id >= c.start && r.id <= c.end);
  assert.equal(originals.length, c.end - c.start + 1);
  for (const r of originals) { assert(!seen.has(r.id)); seen.add(r.id); }
  const preamble = parsed.get(c.start).preamble;
  for (const r of originals) {
    const p = parsed.get(r.id).preamble;
    if (p !== preamble) {
      assert.equal(r.id, 125, 'Unexpected preamble variant');
      assert.equal(p.replace('sulfuric acid to solution 1.', 'sulfuric acid.'), preamble);
    }
  }
  const eligible = originals.filter(r => !excluded.has(r.id));
  if (c.hold) return {...c, original_count: originals.length, eligible_count: eligible.length};
  assert(eligible.length >= 5, 'Only ' + eligible.length + ' verified items in ' + c.id + '; mark it with "hold" in scenarios.json');
  const selected = eligible.map(r => ({id: r.id, hash: sha(seed + '|' + c.id + '|' + r.id)}))
    .sort((a, b) => a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : a.id - b.id)
    .slice(0, 5).map(r => r.id).sort((a, b) => a - b);
  return {...c, original_count: originals.length, eligible_count: eligible.length,
    excluded_ids: originals.filter(r => excluded.has(r.id)).map(r => r.id),
    selected_ids: selected, preamble, items: selected.map(id => parsed.get(id))};
});
assert.equal(seen.size, 200);
const scenarios = all.filter(s => !s.hold);
assert.equal(new Set(scenarios.flatMap(s => s.selected_ids)).size, meta.question_count);
meta.selection_sha256 = sha(JSON.stringify({
  source_sha256: meta.source_sha256, verification_sha256: meta.verification_sha256, seed,
  excluded_ids: meta.excluded_ids, held_scenarios: held.map(c => c.id),
  scenarios: scenarios.map(s => ({id: s.id, source_ids: [s.start, s.end], selected_ids: s.selected_ids}))
}));
const manifest = {
  ...meta, sampling_pool_size: scenarios.reduce((n, s) => n + s.eligible_count, 0),
  exclusions: verification.questions.filter(q => excluded.has(q.id))
    .map(q => ({id: q.id, status: q.status, note: q.note})),
  scenario_notes: 'S13 merges original text groups Q116–124 and Q125 (same sulfuric-acid setup; Q125 adds "to solution 1"). Scenario labels S14–S20 therefore differ from the older 21-stem analysis. Always join results by original question ID.',
  scenarios: scenarios.map(({items, preamble, ...rest}) => rest)
};
const publicData = {
  meta: Object.fromEntries(Object.entries(meta).filter(([k]) => !['excluded_ids', 'sampling', 'seed', 'verification_sha256', 'held_scenarios'].includes(k))),
  scenarios: scenarios.map(s => ({id: s.id, title: s.title, preamble: s.preamble, items: s.items}))
};
function checkNoKeys(value) {
  if (!value || typeof value !== 'object') return;
  for (const [key, val] of Object.entries(value)) {
    assert(!['answer', 'gold', 'truth', 'answer_value', 'answer_tolerance', 'correct_option', 'is_correct', 'author', 'computed', 'status'].includes(key), 'Answer leak: ' + key);
    checkNoKeys(val);
  }
}
checkNoKeys(publicData);
const escapeJson = value => JSON.stringify(value).replace(/</g, '\\u003c').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
const template = fs.readFileSync(path.join(dir, 'template.html'), 'utf8');
const css = fs.readFileSync(path.join(dir, 'style.css'), 'utf8');
const app = fs.readFileSync(path.join(dir, 'app.js'), 'utf8');
const html = template.replace('/* STYLES */', () => css)
  .replace('/* DATA */', () => escapeJson(publicData)).replace('/* APP */', () => app);
assert(!/\/\* (?:STYLES|DATA|APP) \*\//.test(html));
fs.writeFileSync(path.join(dir, 'for_Jerry.html'), html);
fs.writeFileSync(path.join(dir, 'selection_manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
fs.copyFileSync(dbPath, path.join(dir, 'phreeqc.dat'));
console.log(JSON.stringify({
  output: path.join(dir, 'for_Jerry.html'), size_bytes: Buffer.byteLength(html),
  scenarios: scenarios.length, questions: meta.question_count, held: held.map(c => c.id),
  eligible_per_scenario: Object.fromEntries(all.map(s => [s.id, s.eligible_count])),
  excluded: meta.excluded_ids.length, selection_sha256: meta.selection_sha256
}, null, 2));
