// Builds results_macros_verified.tex: all headline numbers on the verified item subset.
// Usage: node neurips_analysis/verified_subset_macros.js <path-to-paper>/results_macros_verified.tex
const fs = require('fs');
const G = '/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github';
const OUT = process.argv[2];
const csv = f => { const [h, ...L] = fs.readFileSync(f, 'utf8').trim().split(/\r?\n/); const k = h.split(','); return L.map(l => Object.fromEntries(l.split(',').map((v, i) => [k[i], v]))); };
const jl = f => fs.readFileSync(f, 'utf8').trim().split('\n').map(JSON.parse);
const rows = jl(G + '/datasets/neurips_dataset_200_clean.jsonl');
const ver = JSON.parse(fs.readFileSync(G + '/human_evaluation/verification.json'));
const st = new Map(ver.questions.map(q => [q.id, q.status]));
const cfg = JSON.parse(fs.readFileSync(G + '/human_evaluation/scenarios.json'));
const scen = id => cfg.find(c => id >= c.start && id <= c.end).id;
const ALL = rows.map(r => r.id), V = ALL.filter(i => st.get(i) === 'verified');
const run = d => { const L = jl(G + '/result/' + d + '/results.jsonl'); if (L.length !== 200) throw new Error(d); return Object.fromEntries(L.map(x => [rows[x.index - 1].id, x.is_correct === true ? 1 : 0])); };

const NAMES = {'Claude Opus 4.6': 'Opus', 'Claude Sonnet 4.6': 'Sonnet', 'GPT-5.4': 'GPTFiveFour', 'GPT-5.2': 'GPTFiveTwo', 'GPT-5.1': 'GPTFiveOne', 'Gemini 3 Flash': 'GeminiThreeFlash'};
const M = {};
for (const r of csv(G + '/neurips_analysis/baseline_snapshot/spliced_per_item_correctness.csv')) {
  const m = NAMES[r.model]; M[m] ??= {};
  for (const c of ['direct', 'cot', 'toc_new', 'raw100k', 'raw500k']) { M[m][c] ??= {}; if (r[c] !== '') M[m][c][+r.item_id] = +r[c]; }
}
M.Opus.direct = run('oneshot/neurips200_clean_oneshot_opus46');
M.Opus.cot = run('oneshot/neurips200_clean_cot_opus46');
M.GeminiThreeFlash = {direct: run('oneshot/neurips200_clean_oneshot_gflash3'), cot: run('oneshot/neurips200_clean_cot_gflash3'),
  toc_new: run('custom/neurips200_clean_agent100k_toc_gflash3'), raw100k: run('custom/neurips200_clean_agent100k_raw100k_gflash3'), raw500k: {}};

const full = (m, c, ids) => ids.every(i => M[m][c][i] !== undefined);
const nCorrect = (m, c, ids) => ids.filter(i => M[m][c][i] === 1).length;
const acc = (m, c, ids) => 100 * nCorrect(m, c, ids) / ids.length;
const wilson = (k, n) => { const z = 1.959964, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [100 * (c - h), 100 * (c + h)]; };
const pct = x => x.toFixed(1) + '\\%';
const pp = x => (x >= 0 ? '$+$' : '$-$') + Math.abs(x).toFixed(1) + '\\,pp';
const lines = ['% Headline results on the verified item subset (generated; do not edit by hand).',
  '% Source: geos_github/human_evaluation/verification.json + existing per-item correctness; no model was re-run.',
  '% Token costs are not item-subset dependent and remain the run-level macros in results_macros.tex.'];
const def = (k, v) => lines.push(`\\newcommand{\\${k}}{${v}}`);
const sanity = {};

// item set
const statusCount = s => ALL.filter(i => st.get(i) === s).length;
const s17 = ALL.filter(i => scen(i) === 'S17' && st.get(i) !== 'verified' && st.get(i) !== 'unstable').length;
def('PoolN', ALL.length); def('VerN', V.length); def('ExclN', ALL.length - V.length);
def('ExclWrongKeyN', statusCount('wrong_key')); def('ExclUnstableN', statusCount('unstable'));
def('ExclWordingN', ALL.length - V.length - statusCount('wrong_key') - statusCount('unstable'));
const setupsWithVerified = new Set(V.map(scen)).size;
def('SetupN', cfg.length); def('VerSetupN', setupsWithVerified);
const keyDist = ['A', 'B', 'C', 'D'].map(k => V.filter(i => rows.find(r => r.id === i).answer === k).length);
def('VerTruthCounts', keyDist.join('/')); def('VerMajority', pct(100 * Math.max(...keyDist) / V.length));

// main table
for (const m of Object.values(NAMES)) for (const [c, tag] of [['direct', 'OneShot'], ['cot', 'CoT'], ['toc_new', 'TOC'], ['raw100k', 'Raw']]) {
  if (!full(m, c, V)) throw new Error(m + ' ' + c);
  const k = nCorrect(m, c, V); def('v' + m + tag, pct(100 * k / V.length));
  const [lo, hi] = wilson(k, V.length); def('v' + m + tag + 'CI', `[${lo.toFixed(1)}, ${hi.toFixed(1)}]`);
  sanity[m + tag] = [acc(m, c, ALL).toFixed(1), acc(m, c, V).toFixed(1)];
}
for (const m of ['GPTFiveFour', 'GPTFiveTwo', 'GPTFiveOne']) def('v' + m + 'RawFiveHundredK', pct(acc(m, 'raw500k', V)));
// derived: CoT minus direct, tool gain over better no-tool baseline, TOC minus raw
const five = ['Opus', 'Sonnet', 'GPTFiveFour', 'GPTFiveTwo', 'GPTFiveOne'];
const cotDelta = Object.fromEntries([...five, 'GeminiThreeFlash'].map(m => [m, acc(m, 'cot', V) - acc(m, 'direct', V)]));
const toolGain = Object.fromEntries(five.map(m => [m, acc(m, 'toc_new', V) - Math.max(acc(m, 'direct', V), acc(m, 'cot', V))]));
const tocRaw = Object.fromEntries([...five, 'GeminiThreeFlash'].map(m => [m, acc(m, 'toc_new', V) - acc(m, 'raw100k', V)]));
for (const m of [...five, 'GeminiThreeFlash']) { def('v' + m + 'CoTMinusDirect', pp(cotDelta[m])); def('v' + m + 'TOCMinusRaw', pp(tocRaw[m])); }
for (const m of five) def('v' + m + 'ToolGain', pp(toolGain[m]));
const gainVals = five.map(m => toolGain[m]);
def('vToolGainMin', Math.min(...gainVals).toFixed(1)); def('vToolGainMax', Math.max(...gainVals).toFixed(1));
const cotAbsMax = Math.max(...five.map(m => Math.abs(cotDelta[m])));
def('vCoTAbsMax', cotAbsMax.toFixed(1));
const gem = 'GeminiThreeFlash';
def('vGeminiThreeFlashTOCMinusBestNoTool', pp(acc(gem, 'toc_new', V) - Math.max(acc(gem, 'direct', V), acc(gem, 'cot', V))));
def('vGeminiThreeFlashRawMinusDirect', pp(acc(gem, 'raw100k', V) - acc(gem, 'direct', V)));
def('vGeminiThreeFlashBestNoTool', pct(Math.max(acc(gem, 'direct', V), acc(gem, 'cot', V))));

// retention
const ret = (m, c, ids) => { let k = 0, g = 0, l = 0; for (const i of ids) { const d = M[m].direct[i], t = M[m][c][i]; if (d && t) k++; else if (!d && t) g++; else if (d && !t) l++; } return {k, g, l, r: 100 * k / (k + l)}; };
const R = {};
for (const m of five) {
  const a = ret(m, 'toc_new', V), b = ret(m, 'raw100k', V); R[m] = {a, b};
  def('v' + m + 'Kept', a.k); def('v' + m + 'Gained', a.g); def('v' + m + 'Lost', a.l); def('v' + m + 'Retention', pct(a.r));
  def('v' + m + 'RawKept', b.k); def('v' + m + 'RawGained', b.g); def('v' + m + 'RawLost', b.l); def('v' + m + 'RawRetention', pct(b.r));
  def('v' + m + 'TOCMinusRawRetention', pp(a.r - b.r));
  const dn = (a.g - a.l) - (b.g - b.l); def('v' + m + 'TOCMinusRawNet', (dn >= 0 ? '$+' : '$-') + Math.abs(dn) + '$');
  def('v' + m + 'LostPct', pct(100 * a.l / (a.k + a.l)));
}
const rets = five.map(m => R[m].a.r), losts = five.map(m => R[m].a.l), gains = five.map(m => R[m].a.g);
def('vRetentionMin', Math.min(...rets).toFixed(1)); def('vRetentionMax', Math.max(...rets).toFixed(1));
def('vLostMin', Math.min(...losts)); def('vLostMax', Math.max(...losts));
def('vGainedMin', Math.min(...gains)); def('vGainedMax', Math.max(...gains));
// lost on flagged (excluded) items, under the same TOC run
for (const m of five) { const ex = ALL.filter(i => st.get(i) !== 'verified'); def('v' + m + 'LostOnExcluded', ex.filter(i => M[m].direct[i] && M[m].toc_new[i] === 0).length); }

// output-size slices (canonical reference sizes + 5-model long-output union)
const sizes = csv(G + '/neurips_analysis/baseline_snapshot/canonical_output_sizes_sonnet_run1.csv');
const small = sizes.filter(r => r.cap_affected === '0').map(r => +r.item_id);
const bin10 = sizes.filter(r => +r.result_out_size_chars <= 10000).map(r => +r.item_id);
const union45 = fs.readFileSync(G + '/neurips_analysis/baseline_snapshot/cap_affected_union45_ids.txt', 'utf8').trim().split(/\s+/).map(Number);
const Vs = V.filter(i => small.includes(i)), Vl = V.filter(i => union45.includes(i)), V10 = V.filter(i => bin10.includes(i));
def('vSmallN', Vs.length); def('vLongN', Vl.length); def('vBinTenKN', V10.length);
for (const m of five) {
  def('v' + m + 'TOCSmall', pct(acc(m, 'toc_new', Vs))); def('v' + m + 'RawSmall', pct(acc(m, 'raw100k', Vs))); def('v' + m + 'TOCMinusRawSmall', pp(acc(m, 'toc_new', Vs) - acc(m, 'raw100k', Vs)));
  def('v' + m + 'TOCLong', pct(acc(m, 'toc_new', Vl))); def('v' + m + 'RawLong', pct(acc(m, 'raw100k', Vl))); def('v' + m + 'TOCMinusRawLong', pp(acc(m, 'toc_new', Vl) - acc(m, 'raw100k', Vl)));
  const b = ret(m, 'toc_new', V10); def('v' + m + 'LostPctBinTenK', pct(100 * b.l / (b.k + b.l)));
}
for (const m of ['GPTFiveFour', 'GPTFiveTwo', 'GPTFiveOne']) {
  def('v' + m + 'RawFiveHundredKLong', pct(acc(m, 'raw500k', Vl)));
  def('v' + m + 'RawFiveHundredKMinusRawLong', pp(acc(m, 'raw500k', Vl) - acc(m, 'raw100k', Vl)));
  let g = 0, l = 0; for (const i of Vl) { if (!M[m].raw100k[i] && M[m].raw500k[i]) g++; if (M[m].raw100k[i] && !M[m].raw500k[i]) l++; }
  def('v' + m + 'RawFiveHundredKLongGain', g); def('v' + m + 'RawFiveHundredKLongLoss', l);
  // run-to-run noise: flips on small-output items, where both windows return the same output
  let fg = 0, fl = 0; for (const i of Vs) { if (!M[m].raw100k[i] && M[m].raw500k[i]) fg++; if (M[m].raw100k[i] && !M[m].raw500k[i]) fl++; }
  def('v' + m + 'RawWindowFlipsSmall', fg + fl);
}

// strict subset: also drop retained items in setups whose text does not fully specify the reference input
const STRICT_DROP = ['S06', 'S07', 'S09'];
const S = V.filter(i => !STRICT_DROP.includes(scen(i)));
def('StrictN', S.length); def('StrictDropItemsN', V.length - S.length);
for (const m of [...five, 'GeminiThreeFlash']) for (const [c, tag] of [['direct', 'OneShot'], ['cot', 'CoT'], ['toc_new', 'TOC'], ['raw100k', 'Raw']]) def('s' + m + tag, pct(acc(m, c, S)));
for (const m of five) { const a = ret(m, 'toc_new', S); def('s' + m + 'Retention', pct(a.r)); def('s' + m + 'Lost', a.l); def('s' + m + 'TOCMinusRaw', pp(acc(m, 'toc_new', S) - acc(m, 'raw100k', S))); }
// full pool, same derived quantities, for the appendix comparison table
for (const m of five) { const a = ret(m, 'toc_new', ALL); def('f' + m + 'Retention', pct(a.r)); def('f' + m + 'Lost', a.l); def('f' + m + 'TOCMinusRaw', pp(acc(m, 'toc_new', ALL) - acc(m, 'raw100k', ALL))); }
for (const m of [...five, 'GeminiThreeFlash']) for (const [c, tag] of [['direct', 'OneShot'], ['cot', 'CoT'], ['toc_new', 'TOC'], ['raw100k', 'Raw']]) def('f' + m + tag, pct(acc(m, c, ALL)));
// overlap of TOC-lost sets on the verified items
const lostSets = five.map(m => V.filter(i => M[m].direct[i] && M[m].toc_new[i] === 0));
const lostUnion = new Set(lostSets.flat());
const lostByCount = [...lostUnion].map(i => lostSets.filter(L => L.includes(i)).length);
def('vLostUnionN', lostUnion.size); def('vLostSingleModelN', lostByCount.filter(c => c === 1).length);
def('vLostThreePlusN', lostByCount.filter(c => c >= 3).length);
console.log('lost overlap: union', lostUnion.size, 'single-model', lostByCount.filter(c => c === 1).length, '>=3 models', lostByCount.filter(c => c >= 3).length, 'all five', lostByCount.filter(c => c === 5).length);

// rebuttal-period controls: generic keyword-search interface (Grep), independent Raw100k repeats,
// and TOC reruns spliced with the 45-item uncapped-TOC fix (same splice as the headline TOC runs)
const fixRun = d => { const L = jl(G + '/result/custom/' + d + '/results.jsonl'); L.forEach((x, k) => { if (x.index !== union45[k]) throw new Error('tocfix order ' + d); }); return Object.fromEntries(L.map(x => [x.index, x.is_correct === true ? 1 : 0])); };
const splice = (r, f) => Object.fromEntries(ALL.map(i => [i, union45.includes(i) ? f[i] : r[i]]));
const accMap = (mp, ids) => 100 * ids.filter(i => mp[i] === 1).length / ids.length;
const RR = {
  Sonnet: ['sonnet46', ['neurips200_clean_rerun_20260504_sonnet_run1_toc_sonnet46', 'neurips200_clean_rerun_20260504_sonnet_run2_toc_sonnet46', 'neurips200_clean_rerun_20260504_sonnet_run3_toc_sonnet46']],
  GPTFiveFour: ['gpt_5_4_', ['neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt_5_4_', 'neurips200_clean_rerun_20260503_alltoc_run2_redo_gpt54_gpt52_toc_gpt_5_4_', 'neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt_5_4_']],
  GPTFiveTwo: ['gpt52', ['neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt52', 'neurips200_clean_rerun_20260503_alltoc_run2_redo_gpt54_gpt52_toc_gpt52', 'neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt52']],
  GPTFiveOne: ['gpt51', ['neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt51', 'neurips200_clean_rerun_20260503_alltoc_run2_toc_gpt51', 'neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt51']]};
for (const [m, [tag, reruns]] of Object.entries(RR)) {
  const f = fixRun('neurips200_tocfix_uncapped_capaffected45_toc_' + tag);
  const R = reruns.map(d => splice(run('custom/' + d), f));
  const accs = [M[m].toc_new, ...R].map(r => accMap(r, V));
  def('v' + m + 'TOCRunMin', pct(Math.min(...accs))); def('v' + m + 'TOCRunMax', pct(Math.max(...accs)));
  const dc = V.filter(i => M[m].direct[i] === 1);
  const lostMaj = dc.filter(i => R.filter(r => r[i] === 0).length >= 2).length;
  const lostAll = dc.filter(i => R.every(r => r[i] === 0)).length;
  def('v' + m + 'LostMajority', lostMaj); def('v' + m + 'LostAllReruns', lostAll);
  def('v' + m + 'RetentionMajority', pct(100 * (dc.length - lostMaj) / dc.length));
}
const GREP = {Sonnet: 'neurips200_grep_sonnet46', GPTFiveTwo: 'neurips200_grep_gpt52', GPTFiveOne: 'neurips200_grep_gpt51'};
for (const [m, d] of Object.entries(GREP)) def('v' + m + 'Grep', pct(accMap(run('custom/' + d), V)));
const REP = {GPTFiveTwo: 'rebuttal_mcq200_raw100k_rerun_gpt52_20260723', GPTFiveOne: 'rebuttal_mcq200_raw100k_rerun_gpt51_20260723'};
for (const [m, d] of Object.entries(REP)) {
  const r = run('custom/' + d); def('v' + m + 'RawRepeat', pct(accMap(r, V)));
  def('v' + m + 'RawRepeatFlips', V.filter(i => r[i] !== M[m].raw100k[i]).length);
}

// independent expert re-solution (human_evaluation/responses, P01)
const resp = JSON.parse(fs.readFileSync(G + '/human_evaluation/responses/participants/P01/phreeqc_P01-Validation_final_2026-10-09T23-03-56-184Z.json'));
const key = new Map(rows.map(r => [r.id, r.answer]));
const committed = resp.responses.filter(r => r.choice !== 'E');
def('HumanN', resp.responses.length); def('HumanScenarioN', new Set(resp.responses.map(r => r.scenario_id)).size);
def('HumanCommitted', committed.length); def('HumanAgree', committed.filter(r => r.choice === key.get(r.question_id)).length);
def('HumanAbstain', resp.responses.length - committed.length);
def('HumanAgreePct', pct(100 * committed.filter(r => r.choice === key.get(r.question_id)).length / committed.length));
def('HumanProblemReports', resp.responses.filter(r => r.problem_reported).length);
def('HumanVersionAbstain', 5);
def('HumanYears', 8);

fs.writeFileSync(OUT, lines.join('\n') + '\n');
console.log('wrote', OUT, lines.length - 3, 'macros');
console.log('sanity 200 vs V:', JSON.stringify(sanity));
console.log('V', V.length, 'setups', setupsWithVerified, 'small', Vs.length, 'long', Vl.length, 'bin10', V10.length);
