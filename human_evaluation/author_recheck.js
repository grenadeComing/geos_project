#!/usr/bin/env node
'use strict';
// Builds for_Sachit.html: the offline page the benchmark author fills in and sends back as JSON.
// Flagged questions are grouped by cause so one decision can settle a whole group.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert/strict');
const dir = __dirname;
const repo = path.dirname(dir);
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const sourceBytes = fs.readFileSync(path.join(repo, 'datasets/neurips_dataset_200_clean.jsonl'));
const rows = sourceBytes.toString('utf8').trim().split(/\r?\n/).map(JSON.parse);
const byId = new Map(rows.map(r => [r.id, r]));
const configs = JSON.parse(fs.readFileSync(path.join(dir, 'scenarios.json'), 'utf8'));
const vBytes = fs.readFileSync(path.join(dir, 'verification.json'));
const v = JSON.parse(vBytes);
assert.equal(v.questions.length, 200);
const check = new Map(v.questions.map(q => [q.id, q]));
const setupOf = new Map(v.scenarios.map(s => [s.id, s]));
const LABEL = {verified: 'Verified', wrong_key: 'Likely wrong key', no_match: 'No option matches', unstable: 'Unstable value', ambiguous: 'Ambiguous or defective text'};
const ALL = ['keep', 'change_key', 'revise', 'drop'];
const flaggedIds = v.questions.filter(q => q.status !== 'verified').map(q => q.id);
const unstableIds = v.questions.filter(q => q.status === 'unstable').map(q => q.id);

const GROUPS = [
  {id: 'wrong-key', title: 'The key looks wrong', ids: [22], options: ['change_key', 'keep', 'revise', 'drop'], suggested: 'change_key',
    explain: 'PHREEQC reproduces a different option than the key, and the result does not change with solver settings.'},
  {id: 'missing-stem', title: 'The question sentence is missing', ids: [4, 37, 78], options: ['revise', 'drop'], suggested: 'revise',
    explain: 'Only the setup and options A–D are in the dataset, so a reader cannot tell what is being asked. Write the question sentence for each, or drop them.'},
  {id: 'moles-molality', title: 'Asks for moles, but the key is the molality', ids: [54, 64, 66], options: ['revise', 'change_key', 'keep', 'drop'], suggested: 'revise',
    explain: 'In these systems the mass of water is not 1 kg (2.26 kg in S06, 1.78 kg in S07), so moles and molality differ. The keyed option equals the molality. Change "moles" to "molality" in the question, or change the options to moles.'},
  {id: 'unstable', title: 'Redox values that PHREEQC does not pin down', ids: unstableIds, options: ['drop', 'revise', 'keep'], suggested: 'drop',
    explain: 'These ask for pe or a value that follows from it (log IAP or SI of H2(g), O2(g), CH4(g), sulfur; H2 molality) in systems with no redox couple to fix it. Changing solver settings moves the value by more than the gap between options: in S14, pe is 10.741 with default settings and −0.920 with a tighter convergence tolerance. Rewording cannot fix this. Drop them, or revise each to ask for a quantity that is well defined.'},
  {id: 's16-location', title: 'S16: a solution property asked "at exchange N"', ids: [151, 153, 159, 160], options: ['revise', 'keep', 'drop'], suggested: 'revise',
    explain: 'Specific conductance, saturation index and log IAP belong to the solution in a cell, not to the exchanger. The key matches when "exchange N" is read as cell N. For Q153, "Na at exchange 1" could also mean NaX on the exchanger, which matches no option. Saying "solution in cell N" removes the doubt.'},
  {id: 's17-units', title: 'S17: concentrations have no units', ids: [163, 164, 165, 167, 168, 169, 170, 171, 172], setups: ['S17'],
    options: ['keep', 'revise', 'drop'], suggested: 'keep', labels: {keep: 'Keep them; the setup fix resolves these'},
    explain: 'With PHREEQC’s default unit (mmol/kgw) no solid solution forms and none of these keys match. With mol/kgw every one of them matches exactly. Adding the unit to the setup text fixes all nine. (Q166, the pe in this scenario, is in the redox group.)'},
  {id: 's01-steps', title: 'S01: reaction-step numbering', ids: [5, 6], setups: ['S01'], options: ['revise', 'change_key', 'keep', 'drop'], suggested: 'revise',
    explain: 'Q5 and Q6 ask about reaction step 47, but their keys match step 48. Q7, also at step 47, matches step 47. Either the questions should say step 48, or the options should be recomputed for step 47.'},
  {id: 'one-off', title: 'Other single issues', ids: [3, 56, 186], options: ALL, perQuestion: true,
    explain: 'Each of these has its own problem, so decide them one by one.'},
  {id: 'setup-text', title: 'Setup text does not match your input', ids: [], setups: ['S05', 'S06', 'S07', 'S09', 'S12'],
    explain: 'The setup text that every question in these scenarios starts with does not describe the input that produced the keys, or leaves out something the answers depend on. A reader who follows the text literally gets different numbers. Most of these questions are otherwise verified, so a corrected setup text would make the whole scenario usable.'}
];
const grouped = GROUPS.flatMap(g => g.ids);
assert.equal(new Set(grouped).size, grouped.length, 'A question is in two groups');
assert.deepEqual([...grouped].sort((a, b) => a - b), flaggedIds, 'Groups must cover every flagged question exactly once');
const setupIds = GROUPS.flatMap(g => g.setups || []);
for (const id of setupIds) assert(setupOf.get(id).issue, id + ' has no recorded issue');

function parts(r) {
  const t = r.question.replace(/\r\n/g, '\n');
  const cut = t.search(/\n\s*\n/);
  const rest = t.slice(cut).trimStart();
  const at = rest.search(/^A\.[ \t]*/m);
  const options = Object.fromEntries([...rest.slice(at).matchAll(/^([A-D])\.[ \t]*(.*)$/gm)].map(m => [m[1], m[2]]));
  assert.deepEqual(Object.keys(options), ['A', 'B', 'C', 'D'], 'Options for Q' + r.id);
  return {preamble: t.slice(0, cut), stem: rest.slice(0, at).trim(), options};
}
const scenarioOf = id => configs.find(c => id >= c.start && id <= c.end);
const items = flaggedIds.map(id => {
  const q = check.get(id), r = byId.get(id), p = parts(r);
  return {id, scenario_id: scenarioOf(id).id, group: GROUPS.find(g => g.ids.includes(id)).id, status: q.status, status_label: LABEL[q.status],
    preamble: p.preamble, stem: p.stem, options: p.options, key: r.answer, matched: q.matched_options || [], computed: q.computed || '', note: q.note};
});
const scenarios = configs.map(c => {
  const s = setupOf.get(c.id);
  const m = /^Sachit input \((test\d+)\)/.exec(s.setup_source);
  return {id: c.id, title: c.title, preamble: parts(byId.get(c.start)).preamble, issue: s.issue, from_test: m ? m[1] : null,
    setup_source: s.setup_source, input: s.input, needs_input: !m,
    questions: !setupIds.includes(c.id) ? [] : v.questions.filter(q => q.id >= c.start && q.id <= c.end).map(q => {
      const r = byId.get(q.id), p = parts(r);
      return {id: q.id, status: q.status, status_label: LABEL[q.status], stem: p.stem, options: p.options, key: r.answer};
    })};
});
const payload = {meta: {dataset_sha256: sha(sourceBytes), verification_sha256: sha(vBytes), date: v.date},
  groups: GROUPS.map(g => ({setups: [], labels: {}, perQuestion: false, ...g})), scenarios, items};
const esc = x => JSON.stringify(x).replace(/</g, '\\u003c').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
const h = x => String(x).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const css = fs.readFileSync(path.join(dir, 'recheck.css'), 'utf8');
const app = fs.readFileSync(path.join(dir, 'recheck_app.js'), 'utf8');
const nDecisions = GROUPS.filter(g => g.ids.length && !g.perQuestion).length + GROUPS.filter(g => g.perQuestion).reduce((n, g) => n + g.ids.length, 0) + setupIds.length;
const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>PHREEQC-MCQ-200 Recheck</title>
<style>${css}</style>
</head>
<body>
<header class="intro">
  <div class="eyebrow">Answer-key recheck · ${h(v.date)}</div>
  <h1>PHREEQC-MCQ-200 recheck</h1>
  <p>Sachit, we re-ran your 200 questions in PHREEQC ${h(v.phreeqc_version)} with the frozen <code>phreeqc.dat</code>. For ${200 - flaggedIds.length} of them we reproduced your key exactly; they are not listed here. The other ${flaggedIds.length}, and ${setupIds.length} scenario setups, are grouped below by the kind of problem, which comes to about ${nDecisions} decisions.</p>
  <ul class="how">
    <li>Each group has one decision that applies to all its questions. Under any question you can choose <b>Use a different decision</b> to override it.</li>
    <li>Every question is shown in full, as you wrote it: setup, question, and A–D with your key marked. Our PHREEQC value and what we found are in the box under it.</li>
    <li>If you change a key or revise a question, the fields to fill in appear under that question.</li>
    <li>Your work saves in this browser as you go. Press <b>Download my review</b> when you are done (or to pause), and send us the file. This page makes no network requests and sends nothing by itself.</li>
  </ul>
  <p class="muted">Our numbers come from our own inputs, which you can open and copy on each scenario. For S09 and S11–S20 we did not have your input files, so please check our setup against yours and attach your files at the bottom if you still have them.</p>
  <div class="fields">
    <label style="flex:1 1 260px">Your name<input type="text" id="reviewer" maxlength="120" autocomplete="name"></label>
    <label class="file-button">Restore a downloaded review<input type="file" id="restore" accept=".json,application/json"></label>
  </div>
</header>
<div class="toolbar"><div class="toolbar-inner">
  <span class="progress" id="progress-count"></span>
  <div class="bar" aria-hidden="true"><span id="progress-bar"></span></div>
  <label class="toggle"><input type="checkbox" id="only-open"> Show only what still needs a decision</label>
  <div class="actions"><button type="button" class="primary" id="download">Download my review</button></div>
  <p class="status-line" id="status-line" role="status" aria-live="polite"></p>
</div></div>
<main>
  <nav class="toc" aria-label="Groups"><ol id="toc"></ol></nav>
  <div id="groups" class="groups"></div>
  <section aria-labelledby="h-inputs" class="group">
    <h2 id="h-inputs">Your PHREEQC input files</h2>
    <p class="section-intro">Optional but very helpful. Attached files are stored inside the downloaded file (up to 1 MB each).</p>
    <div id="input-rows" class="inputs"></div>
  </section>
</main>
<noscript>This page needs JavaScript. It makes no network requests.</noscript>
<script id="recheck-data" type="application/json">${esc(payload)}</script>
<script>${app}</script>
</body>
</html>
`;
fs.writeFileSync(path.join(dir, 'for_Sachit.html'), html);
console.log(JSON.stringify({output: path.join(dir, 'for_Sachit.html'), questions: items.length, setups: setupIds.length,
  groups: GROUPS.map(g => g.id + ':' + g.ids.length), decisions: nDecisions, bytes: Buffer.byteLength(html)}, null, 2));
