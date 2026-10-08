#!/usr/bin/env node
'use strict';
// Builds author_review.html: the answer-key review sent to the benchmark author.
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const dir = __dirname;
const repo = path.dirname(dir);
const rows = fs.readFileSync(path.join(repo, 'datasets/neurips_dataset_200_clean.jsonl'), 'utf8')
  .trim().split(/\r?\n/).map(JSON.parse);
const byId = new Map(rows.map(r => [r.id, r]));
const configs = JSON.parse(fs.readFileSync(path.join(dir, 'scenarios.json'), 'utf8'));
const v = JSON.parse(fs.readFileSync(path.join(dir, 'verification.json'), 'utf8'));
const checks = new Map(v.questions.map(q => [q.id, q]));
assert.equal(checks.size, 200);
const scenarioOf = id => configs.find(c => id >= c.start && id <= c.end);
const setupOf = new Map(v.scenarios.map(s => [s.id, s]));

const h = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const GROUPS = [
  {key: 'wrong', label: 'Likely wrong key', statuses: ['wrong_key'],
    intro: 'PHREEQC reproduces a different option than the key, and the result is stable. These most likely need the answer letter (or the option values) corrected.'},
  {key: 'suspicious', label: 'Suspicious', statuses: ['no_match', 'unstable', 'ambiguous'],
    intro: 'The key could not be confirmed: no option matches our run, the value is numerically unstable, or the question text is ambiguous or defective. Each needs a decision from you: fix the wording, fix the options, or drop the item.'},
  {key: 'unverified', label: 'Could not verify', statuses: ['not_verified'],
    intro: 'We could not reproduce the setup well enough to check these. They are not known to be wrong. Your original input files would settle them.'},
  {key: 'verified', label: 'Verified', statuses: ['verified'],
    intro: 'Our run reproduces the keyed option exactly and uniquely. No action needed.'}
];
const STATUS_LABEL = {wrong_key: 'wrong key', no_match: 'no option matches', unstable: 'unstable',
  ambiguous: 'ambiguous text', not_verified: 'not verified', verified: 'verified'};
const groupOf = status => GROUPS.find(g => g.statuses.includes(status));
for (const q of v.questions) assert(groupOf(q.status), 'Unknown status for Q' + q.id + ': ' + q.status);

function parts(r) {
  const text = r.question.replace(/\r\n/g, '\n');
  const cut = text.search(/\n\s*\n/);
  const rest = text.slice(cut).trimStart();
  const at = rest.search(/^A\.[ \t]*/m);
  const options = Object.fromEntries([...rest.slice(at).matchAll(/^([A-D])\.[ \t]*(.*)$/gm)].map(m => [m[1], m[2]]));
  return {stem: rest.slice(0, at).trim(), options};
}
function optionList(r, q) {
  const {options} = parts(r);
  return '<ol class="opts">' + Object.entries(options).map(([k, val]) => {
    const cls = [k === r.answer ? 'key' : '', (q.matched_options || []).includes(k) ? 'match' : ''].filter(Boolean).join(' ');
    return '<li' + (cls ? ' class="' + cls + '"' : '') + '><span class="letter">' + k + '</span><span class="num">' + h(val) + '</span>' +
      (k === r.answer ? '<span class="tag">key</span>' : '') +
      ((q.matched_options || []).includes(k) ? '<span class="tag run">PHREEQC</span>' : '') + '</li>';
  }).join('') + '</ol>';
}
function card(q) {
  const r = byId.get(q.id), s = scenarioOf(q.id), {stem} = parts(r);
  return '<article class="item" id="q' + q.id + '"><header><span class="qid">Q' + q.id + '</span><span class="scn">' + s.id + ' · ' + h(s.title) + '</span>' +
    '<span class="pill ' + q.status + '">' + STATUS_LABEL[q.status] + '</span></header>' +
    '<p class="stem">' + (stem ? h(stem) : '<em>No question stem in the dataset. Only the shared setup and options A–D are present.</em>') + '</p>' +
    '<div class="grid"><div>' + optionList(r, q) + '</div><dl>' +
    '<dt>Our PHREEQC value</dt><dd class="num">' + h(q.computed || '—') + '</dd>' +
    '<dt>Finding</dt><dd>' + h(q.note) + '</dd></dl></div></article>';
}

const counts = Object.fromEntries(GROUPS.map(g => [g.key, v.questions.filter(q => g.statuses.includes(q.status)).length]));
const SHOWN = GROUPS.filter(g => counts[g.key] > 0);
const matrix = configs.map(c => {
  const qs = v.questions.filter(q => q.id >= c.start && q.id <= c.end);
  const cells = SHOWN.map(g => {
    const ids = qs.filter(q => g.statuses.includes(q.status)).map(q => q.id);
    return '<td class="num ' + g.key + (ids.length ? '' : ' zero') + '">' + ids.length + '</td>';
  }).join('');
  const flagged = qs.filter(q => q.status !== 'verified').map(q => '<a href="#q' + q.id + '">' + q.id + '</a>').join(', ');
  return '<tr><td class="sid">' + c.id + '</td><td>' + h(c.title) + '<span class="range">Q' + c.start + '–' + c.end + '</span></td>' + cells +
    '<td class="flag">' + (flagged || '—') + '</td></tr>';
}).join('');

const sections = SHOWN.map(g => {
  const qs = v.questions.filter(q => g.statuses.includes(q.status)).sort((a, b) => a.id - b.id);
  if (g.key === 'verified') {
    const lists = configs.map(c => {
      const ids = qs.filter(q => q.id >= c.start && q.id <= c.end).map(q => q.id);
      return '<li><span class="sid">' + c.id + '</span> ' + (ids.length ? ids.join(', ') : '—') + '</li>';
    }).join('');
    return '<section id="verified"><h2>' + g.label + ' <span class="count">' + qs.length + '</span></h2><p class="intro">' + g.intro + '</p><ul class="verified-list">' + lists + '</ul></section>';
  }
  return '<section id="' + g.key + '"><h2>' + g.label + ' <span class="count">' + qs.length + '</span></h2><p class="intro">' + g.intro + '</p>' +
    (qs.length ? qs.map(card).join('') : '<p class="intro">None.</p>') + '</section>';
}).join('');

const scenarioIssues = configs.filter(c => (setupOf.get(c.id) || {}).issue).map(c =>
  '<li><span class="sid">' + c.id + '</span> <b>' + h(c.title) + '.</b> ' + h(setupOf.get(c.id).issue) + '</li>').join('');
const inputs = configs.map(c => {
  const s = setupOf.get(c.id) || {};
  return '<details><summary><span class="sid">' + c.id + '</span> ' + h(c.title) + ' <span class="src">' + h(s.setup_source || '') + '</span></summary>' +
    (s.setup_notes ? '<p class="intro">' + h(s.setup_notes) + '</p>' : '') +
    (s.input ? '<pre>' + h(s.input) + '</pre>' : '') + '</details>';
}).join('');

const css = `
:root{--paper:#f5f7f6;--surface:#ffffff;--ink:#1b2925;--muted:#5a6964;--line:#d7dedb;--accent:#1f6f63;--accent-soft:#e2efec;
--wrong:#b3261e;--wrong-bg:#fbe9e7;--susp:#9a5b00;--susp-bg:#fdf0dc;--unv:#55598a;--unv-bg:#ebecf6;--ok:#2e7a4d;--ok-bg:#e3f2e8;
--sans:"IBM Plex Sans",-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--paper:#121816;--surface:#1a2220;--ink:#e3ebe8;--muted:#9aaaa4;--line:#2e3a37;--accent:#6cc2b3;--accent-soft:#1d3531;
--wrong:#ff8a80;--wrong-bg:#3a1c1a;--susp:#f0b35c;--susp-bg:#36280f;--unv:#aeb2e6;--unv-bg:#23253a;--ok:#7fd29f;--ok-bg:#15301f}}
:root[data-theme="dark"]{color-scheme:dark;--paper:#121816;--surface:#1a2220;--ink:#e3ebe8;--muted:#9aaaa4;--line:#2e3a37;--accent:#6cc2b3;--accent-soft:#1d3531;
--wrong:#ff8a80;--wrong-bg:#3a1c1a;--susp:#f0b35c;--susp-bg:#36280f;--unv:#aeb2e6;--unv-bg:#23253a;--ok:#7fd29f;--ok-bg:#15301f}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 var(--sans);padding-inline:16px;padding-block:32px 64px}
main{max-width:1040px;margin:0 auto;display:grid;gap:40px}
h1,h2,h3{margin:0;text-wrap:balance;line-height:1.25}h1{font-size:1.9rem;font-weight:600}h2{font-size:1.35rem;font-weight:600;display:flex;align-items:baseline;gap:10px}
p{margin:0}.lede{max-width:68ch;display:grid;gap:10px}.eyebrow{font:600 .72rem/1 var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}
.meta{font:.8rem/1.5 var(--mono);color:var(--muted);overflow-wrap:anywhere}
.totals{display:flex;flex-wrap:wrap;gap:8px}.totals a{display:flex;gap:8px;align-items:baseline;padding:8px 14px;border-radius:999px;text-decoration:none;font-weight:500}
.totals b{font:600 1.05rem var(--mono)}
.t-wrong{background:var(--wrong-bg);color:var(--wrong)}.t-suspicious{background:var(--susp-bg);color:var(--susp)}.t-unverified{background:var(--unv-bg);color:var(--unv)}.t-verified{background:var(--ok-bg);color:var(--ok)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.88rem;min-width:720px}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}tr:last-child td{border-bottom:0}
td.num{font-family:var(--mono);text-align:right;font-weight:600;width:84px}td.zero{color:var(--line);font-weight:400}
td.wrong:not(.zero){color:var(--wrong)}td.suspicious:not(.zero){color:var(--susp)}td.unverified:not(.zero){color:var(--unv)}td.verified:not(.zero){color:var(--ok)}
.sid{font:600 .82rem var(--mono);color:var(--accent)}.range{display:block;font:.75rem var(--mono);color:var(--muted)}td.flag{font-family:var(--mono);font-size:.8rem}
a{color:var(--accent)}a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
section{display:grid;gap:14px}.count{font:600 .9rem var(--mono);color:var(--muted)}.intro{color:var(--muted);max-width:72ch}
.item{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px 18px;display:grid;gap:10px}
.item header{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline}.qid{font:600 1rem var(--mono)}.scn{color:var(--muted);font-size:.85rem;flex:1}
.pill{font:600 .72rem/1 var(--sans);letter-spacing:.04em;text-transform:uppercase;padding:5px 9px;border-radius:999px}
.pill.wrong_key{background:var(--wrong-bg);color:var(--wrong)}.pill.no_match,.pill.unstable,.pill.ambiguous{background:var(--susp-bg);color:var(--susp)}
.pill.not_verified{background:var(--unv-bg);color:var(--unv)}.pill.verified{background:var(--ok-bg);color:var(--ok)}
.stem{font-weight:500}.grid{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:18px}
.opts{list-style:none;margin:0;padding:0;display:grid;gap:4px}.opts li{display:flex;gap:10px;align-items:baseline;padding:4px 8px;border-radius:5px}
.opts li.key{background:var(--accent-soft)}.opts li.match{box-shadow:inset 0 0 0 1.5px var(--ok)}
.letter{font:600 .85rem var(--mono);color:var(--muted);width:1.2em}.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.tag{font:600 .66rem var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}.tag.run{color:var(--ok)}
dl{margin:0;display:grid;gap:4px}dt{font:600 .72rem var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}dd{margin:0 0 8px;overflow-wrap:anywhere}
.verified-list{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px 18px;font:.85rem var(--mono)}
details{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 14px}details+details{margin-top:6px}
summary{cursor:pointer;font-weight:500}.src{color:var(--muted);font-size:.8rem;margin-left:6px}
pre{font:.78rem/1.5 var(--mono);background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:12px;overflow-x:auto;margin:10px 0 4px}
.issues{list-style:none;margin:0;padding:0;display:grid;gap:10px;max-width:80ch}.issues li{padding-left:14px;border-left:3px solid var(--susp)}
ol.asks{margin:0;padding-left:1.3em;display:grid;gap:6px;max-width:72ch}
@media (max-width:640px){.grid{grid-template-columns:1fr}h1{font-size:1.5rem}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}`;

const page = '<title>PHREEQC-MCQ-200 Key Review</title>\n' +
  '<meta name="viewport" content="width=device-width,initial-scale=1">\n' +
  '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>' +
  '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">\n' +
  '<style>' + css + '</style>\n<main>' +
  '<header class="lede"><div class="eyebrow">Answer-key review · ' + h(v.date) + '</div><h1>PHREEQC-MCQ-200 answer-key review</h1>' +
  '<p>Sachit, we re-ran every scenario in your 200-question set with PHREEQC ' + h(v.phreeqc_version) + ' and the same frozen <code>phreeqc.dat</code> used for the benchmark experiments, then compared the printed value for each question with its four options. This page lists what we found so you can fix the key. We have not changed the dataset.</p>' +
  '<p>' + h(v.method) + '</p>' +
  '<p class="meta">database SHA-256 ' + h(v.database_sha256) + '<br>dataset ' + h(v.source_file) + ' SHA-256 ' + h(v.source_sha256) + '</p></header>' +
  '<nav class="totals" aria-label="Totals">' + SHOWN.map(g => '<a class="t-' + g.key + '" href="#' + g.key + '"><b>' + counts[g.key] + '</b>' + g.label + '</a>').join('') + '</nav>' +
  '<section><h2>By scenario</h2><div class="wrap"><table><thead><tr><th>ID</th><th>Scenario</th>' + SHOWN.map(g => '<th>' + g.label + '</th>').join('') + '<th>Questions to check</th></tr></thead><tbody>' + matrix + '</tbody></table></div></section>' +
  '<section><h2>What we need from you</h2><ol class="asks">' +
  '<li>For each <b>likely wrong</b> item, confirm and send the corrected answer letter, or tell us where our setup differs from yours.</li>' +
  '<li>For each <b>suspicious</b> item, fix the wording or options, or mark it to be dropped. Unstable pe values cannot be fixed by rewording; they should be dropped or re-asked as a quantity that a redox couple actually controls.</li>' +
  '<li>Check the <b>scenario-level issues</b>, especially S06, S07, S09 and S17, where the text does not describe the input that produced the key.</li>' +
  '<li>Send your PHREEQC input files for S09 and S11–S20 if you still have them. We only had your inputs for S01–S08 and S10, so the others were checked against our own literal reading of the text. Please compare those inputs (under <a href="#inputs">Inputs we ran</a>) with your setup rather than taking our values as correct.</li>' +
  '<li>Please send corrections as a new file version rather than editing the current one in place, so results on the old version remain traceable.</li></ol></section>' +
  '<section id="scenario-issues"><h2>Scenario-level issues</h2><p class="intro">These affect every question in a scenario, including items whose key we could reproduce. A participant who follows the text literally can get different numbers.</p><ul class="issues">' + scenarioIssues + '</ul></section>' +
  sections +
  '<section id="inputs"><h2>Inputs we ran</h2><p class="intro">One input per scenario. Expand a scenario to see the input and any interpretation we had to choose.</p><div>' + inputs + '</div></section>' +
  '</main>\n';
fs.writeFileSync(path.join(dir, 'author_review.html'), page);
console.log(JSON.stringify({output: path.join(dir, 'author_review.html'), counts}, null, 2));
