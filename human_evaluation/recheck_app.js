(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('recheck-data').textContent);
  const $ = id => document.getElementById(id);
  const storageKey = 'phreeqc-author-recheck-v2:' + data.meta.verification_sha256;
  const LETTERS = ['A', 'B', 'C', 'D'];
  const DECISIONS = {keep: 'Keep as is', change_key: 'Change the key', revise: 'Revise the question', drop: 'Drop'};
  const SETUP_DECISIONS = {revise_setup: 'Revise the setup text', text_ok: 'Setup text is fine as is'};
  const MAX_FILE = 1024 * 1024;
  const scenarioById = new Map(data.scenarios.map(s => [s.id, s]));
  const groupById = new Map(data.groups.map(g => [g.id, g]));
  const setupIds = data.groups.flatMap(g => g.setups);
  let storageOK = true;
  let state = fresh();

  function fresh() {
    return {
      reviewer: '', updated_at: null,
      groups: Object.fromEntries(data.groups.map(g => [g.id, {decision: '', comment: ''}])),
      questions: Object.fromEntries(data.items.map(q => [q.id, {
        override: false, decision: '', new_key: '', stem: q.stem, options: {...q.options}, revised_key: q.key, comment: ''
      }])),
      setups: Object.fromEntries(setupIds.map(id => [id, {decision: '', setup: scenarioById.get(id).preamble, comment: ''}])),
      inputs: Object.fromEntries(data.scenarios.map(s => [s.id, []]))
    };
  }
  const str = (v, max, what) => {
    if (typeof v !== 'string' || v.length > max) throw new Error('Invalid ' + what + '.');
    return v;
  };

  // A question follows its group's decision unless it is overridden or the group is decided per question.
  function ownDecision(q) { return groupById.get(q.group).perQuestion || state.questions[q.id].override; }
  function effective(q) { return ownDecision(q) ? state.questions[q.id].decision : state.groups[q.group].decision; }
  function questionDone(q) {
    const a = state.questions[q.id], d = effective(q);
    if (d === 'keep' || d === 'drop') return true;
    if (d === 'change_key') return LETTERS.includes(a.new_key) && a.new_key !== q.key;
    if (d === 'revise') return Boolean(a.stem.trim()) && LETTERS.every(k => a.options[k].trim()) && LETTERS.includes(a.revised_key);
    return false;
  }
  function setupDone(id) {
    const a = state.setups[id];
    return a.decision === 'text_ok' || (a.decision === 'revise_setup' && Boolean(a.setup.trim()));
  }

  function record() {
    return {
      schema: 'phreeqc-author-recheck', schema_version: 2,
      dataset_sha256: data.meta.dataset_sha256, verification_sha256: data.meta.verification_sha256,
      reviewer: state.reviewer.trim(), updated_at: state.updated_at, exported_at: new Date().toISOString(),
      settled: {questions: data.items.filter(questionDone).length, of_questions: data.items.length,
        setups: setupIds.filter(setupDone).length, of_setups: setupIds.length},
      groups: data.groups.map(g => ({id: g.id, title: g.title, question_ids: g.ids, ...state.groups[g.id]})),
      questions: data.items.map(q => {
        const a = state.questions[q.id];
        return {id: q.id, scenario_id: q.scenario_id, group: q.group, status: q.status, original_key: q.key,
          decision: effective(q), decided_by: ownDecision(q) ? 'question' : 'group', complete: questionDone(q),
          override: a.override, own_decision: a.decision, new_key: a.new_key, stem: a.stem, options: a.options,
          revised_key: a.revised_key, comment: a.comment};
      }),
      scenarios: setupIds.map(id => ({id, complete: setupDone(id), original_setup: scenarioById.get(id).preamble, ...state.setups[id]})),
      inputs: data.scenarios.flatMap(s => state.inputs[s.id].map(f => ({scenario_id: s.id, ...f})))
    };
  }
  function fromRecord(rec) {
    if (!rec || rec.schema !== 'phreeqc-author-recheck' || rec.schema_version !== 2) throw new Error('This is not a review file from this page.');
    if (rec.verification_sha256 !== data.meta.verification_sha256 || rec.dataset_sha256 !== data.meta.dataset_sha256) {
      throw new Error('This file belongs to a different version of the review.');
    }
    const next = fresh();
    next.reviewer = str(rec.reviewer, 120, 'reviewer name');
    next.updated_at = rec.updated_at || null;
    for (const g of data.groups) {
      const r = (rec.groups || []).find(x => x.id === g.id);
      if (!r) throw new Error('Group "' + g.title + '" is missing from the file.');
      if (r.decision !== '' && !g.options.includes(r.decision)) throw new Error('Invalid decision for group "' + g.title + '".');
      next.groups[g.id] = {decision: r.decision, comment: str(r.comment, 8000, 'comment')};
    }
    for (const q of data.items) {
      const r = (rec.questions || []).find(x => x.id === q.id);
      if (!r) throw new Error('Q' + q.id + ' is missing from the file.');
      if (typeof r.override !== 'boolean') throw new Error('Invalid override for Q' + q.id + '.');
      if (r.own_decision !== '' && !Object.hasOwn(DECISIONS, r.own_decision)) throw new Error('Invalid decision for Q' + q.id + '.');
      if (r.new_key !== '' && !LETTERS.includes(r.new_key)) throw new Error('Invalid key for Q' + q.id + '.');
      if (!LETTERS.includes(r.revised_key)) throw new Error('Invalid revised key for Q' + q.id + '.');
      next.questions[q.id] = {override: r.override, decision: r.own_decision, new_key: r.new_key, stem: str(r.stem, 4000, 'question text'),
        options: Object.fromEntries(LETTERS.map(k => [k, str((r.options || {})[k], 1000, 'option')])),
        revised_key: r.revised_key, comment: str(r.comment, 8000, 'comment')};
    }
    for (const id of setupIds) {
      const r = (rec.scenarios || []).find(x => x.id === id);
      if (!r) throw new Error(id + ' is missing from the file.');
      if (r.decision !== '' && !Object.hasOwn(SETUP_DECISIONS, r.decision)) throw new Error('Invalid decision for ' + id + '.');
      next.setups[id] = {decision: r.decision, setup: str(r.setup, 8000, 'setup text'), comment: str(r.comment, 8000, 'comment')};
    }
    for (const f of rec.inputs || []) {
      if (!next.inputs[f.scenario_id]) throw new Error('Unknown scenario in attached files.');
      next.inputs[f.scenario_id].push({filename: str(f.filename, 200, 'file name'), text: str(f.text, MAX_FILE, 'file content')});
    }
    return next;
  }

  const refreshers = [];
  function refreshAll() { refreshers.forEach(f => f()); progress(); }
  function save() {
    state.updated_at = new Date().toISOString();
    try { localStorage.setItem(storageKey, JSON.stringify(record())); storageOK = true; } catch (_) { storageOK = false; }
    status();
  }
  function changed() { save(); refreshAll(); }
  function status(message) {
    const el = $('status-line');
    el.classList.toggle('error', !storageOK);
    el.textContent = message || (!storageOK ? 'Browser storage is unavailable. Download your review before closing this page.' :
      state.updated_at ? 'Saved in this browser at ' + new Date(state.updated_at).toLocaleTimeString() + '. Download the file to send it back.' :
      'Your work saves in this browser as you go. Download the file to send it back.');
  }

  function el(tag, cls, content) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (content !== undefined) n.textContent = content;
    return n;
  }
  function labelled(text, control, cls) { const l = el('label', cls); l.append(el('span', 'field-label', text), control); return l; }
  function textInput(value, onInput, id) {
    const i = el('input', 'mono'); i.type = 'text'; i.value = value; if (id) i.id = id;
    i.addEventListener('input', () => onInput(i.value)); return i;
  }
  function area(value, rows, onInput, id) {
    const t = el('textarea'); t.rows = rows; t.value = value; if (id) t.id = id;
    t.addEventListener('input', () => onInput(t.value)); return t;
  }
  function keySelect(value, onChange, id) {
    const s = el('select', 'small'); s.id = id;
    s.append(new Option('Choose…', ''));
    LETTERS.forEach(k => s.append(new Option(k, k)));
    s.value = value; s.addEventListener('change', () => onChange(s.value)); return s;
  }
  function radios(name, entries, value, onChange, suggested) {
    const box = el('div', 'choices');
    for (const [v, label] of entries) {
      const l = el('label'); const r = el('input'); r.type = 'radio'; r.name = name; r.value = v; r.id = name + '-' + v;
      r.checked = value === v; r.addEventListener('change', () => onChange(v));
      l.append(r, el('span', '', label));
      if (v === suggested) l.append(el('span', 'suggest', 'suggested'));
      box.append(l);
    }
    return box;
  }
  function badge() { return el('span', 'pill todo', 'To decide'); }
  function setBadge(b, done, partial) {
    b.className = 'pill ' + (done ? 'done' : 'todo');
    b.textContent = done ? 'Settled' : partial ? 'Needs input' : 'To decide';
  }
  function stemLine(stem) {
    return stem ? el('p', 'stem pre', stem) : el('p', 'stem missing', 'No question sentence. Only the setup and options A–D are in the dataset.');
  }
  function optionList(options, key, matched) {
    const ol = el('ol', 'opts');
    for (const k of LETTERS) {
      const li = el('li', [k === key ? 'key' : '', matched.includes(k) ? 'match' : ''].join(' ').trim());
      li.append(el('span', 'letter', k), el('span', 'pre', options[k]));
      if (k === key) li.append(el('span', 'tag', 'your key'));
      if (matched.includes(k)) li.append(el('span', 'tag run', 'our run'));
      ol.append(li);
    }
    return ol;
  }
  function scenarioLabel(s) {
    const box = el('div', 'scenario-head');
    box.append(el('span', 'sid', s.id), el('span', 'stitle', s.title));
    if (s.from_test) box.append(el('span', 'from', 'from your ' + s.from_test));
    return box;
  }
  function copyBlock(title, content) {
    const d = el('details', 'reveal');
    const sum = el('summary', '', title);
    const pre = el('pre', '', content);
    const copy = el('button', 'copy', 'Copy'); copy.type = 'button';
    copy.addEventListener('click', () => {
      const done = () => { copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = 'Copy'; }, 1500); };
      const select = () => {
        const r = document.createRange(); r.selectNodeContents(pre);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        copy.textContent = 'Selected: press Ctrl/Cmd+C';
      };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(content).then(done, select); else select();
    });
    d.append(sum, copy, pre);
    return d;
  }

  function questionBlock(q, group) {
    const a = () => state.questions[q.id];
    const box = el('article', 'qblock'); box.id = 'q' + q.id;
    const head = el('div', 'card-head');
    const b = badge();
    head.append(el('span', 'qid', 'Q' + q.id), el('span', 'pill ' + q.status, q.status_label), el('span', 'grow'), b);
    const finding = el('dl', 'finding');
    finding.append(el('dt', '', 'Our PHREEQC value'), el('dd', 'mono', q.computed || '—'), el('dt', '', 'What we found'), el('dd', '', q.note));

    const own = el('div', 'own');
    let overrideBox = null;
    if (!group.perQuestion) {
      overrideBox = el('label', 'override');
      const cb = el('input'); cb.type = 'checkbox'; cb.id = 'override-' + q.id; cb.checked = a().override;
      cb.addEventListener('change', () => { a().override = cb.checked; changed(); });
      overrideBox.append(cb, el('span', '', 'Use a different decision for this question'));
    }
    own.append(radios('decision-' + q.id, Object.entries(DECISIONS), a().decision, v => { a().decision = v; changed(); }));

    const keyPanel = el('div', 'panel');
    keyPanel.append(labelled('Correct answer for Q' + q.id, keySelect(a().new_key, v => { a().new_key = v; changed(); }, 'newkey-' + q.id)));
    const revisePanel = el('div', 'panel');
    revisePanel.append(labelled('Question sentence', area(a().stem, 2, v => { a().stem = v; changed(); }, 'stem-' + q.id)));
    const grid = el('div', 'opt-grid');
    for (const k of LETTERS) grid.append(labelled('Option ' + k, textInput(a().options[k], v => { a().options[k] = v; changed(); }, 'opt-' + q.id + '-' + k)));
    revisePanel.append(grid, labelled('Correct answer after revision', keySelect(a().revised_key, v => { a().revised_key = v; changed(); }, 'revkey-' + q.id)));
    const note = el('details', 'reveal');
    note.append(el('summary', '', 'Comment on this question'), area(a().comment, 2, v => { a().comment = v; save(); }, 'comment-' + q.id));
    note.open = Boolean(a().comment);

    box.append(head, stemLine(q.stem), optionList(q.options, q.key, q.matched), finding);
    if (overrideBox) box.append(overrideBox);
    box.append(own, keyPanel, revisePanel, note);
    refreshers.push(() => {
      own.hidden = !ownDecision(q);
      const d = effective(q);
      keyPanel.hidden = d !== 'change_key';
      revisePanel.hidden = d !== 'revise';
      const done = questionDone(q);
      setBadge(b, done, Boolean(d));
      box.classList.toggle('done', done);
      box.hidden = $('only-open').checked && done;
    });
    return box;
  }

  function setupPanel(id) {
    const s = scenarioById.get(id), a = () => state.setups[id];
    const box = el('article', 'setup-card'); box.id = 'setup-' + id;
    const head = el('div', 'card-head'); const b = badge();
    head.append(scenarioLabel(s), el('span', 'grow'), b);
    const current = el('div'); current.append(el('span', 'field-label', 'Setup text as written'), el('p', 'setup-text pre', s.preamble));
    const list = el('details', 'reveal');
    list.append(el('summary', '', 'Show all ' + s.questions.length + ' questions in ' + id));
    const rows = el('div', 'qlist');
    for (const q of s.questions) {
      const row = el('div', 'qrow'); const top = el('div', 'card-head');
      const label = q.status === 'verified' ? el('span', 'qid', 'Q' + q.id) : el('a', 'qid', 'Q' + q.id);
      if (q.status !== 'verified') label.href = '#q' + q.id;
      top.append(label, el('span', 'pill ' + q.status, q.status_label));
      row.append(top, stemLine(q.stem), optionList(q.options, q.key, [])); rows.append(row);
    }
    list.append(rows);
    const fs = el('fieldset'); fs.append(el('legend', '', 'Decision for the ' + id + ' setup'));
    const panel = el('div', 'panel');
    panel.append(labelled('Revised setup text', area(a().setup, 5, v => { a().setup = v; changed(); }, 'setuptext-' + id)));
    fs.append(radios('setup-' + id, Object.entries(SETUP_DECISIONS), a().decision, v => { a().decision = v; changed(); }), panel);
    const note = el('details', 'reveal');
    note.append(el('summary', '', 'Comment on this setup'), area(a().comment, 2, v => { a().comment = v; save(); }));
    note.open = Boolean(a().comment);
    box.append(head, el('p', 'issue', s.issue), current, list, copyBlock('Input we ran (' + s.setup_source + ')', s.input), fs, note);
    refreshers.push(() => {
      panel.hidden = a().decision !== 'revise_setup';
      const done = setupDone(id);
      setBadge(b, done, Boolean(a().decision));
      box.classList.toggle('done', done);
      box.hidden = $('only-open').checked && done;
    });
    return box;
  }

  function groupCard(g, index) {
    const section = el('section', 'group'); section.id = 'group-' + g.id;
    const head = el('div', 'group-head');
    const count = el('span', 'count');
    head.append(el('span', 'gnum', String(index + 1)), el('h2', 'grow', g.title), count);
    section.append(head, el('p', 'explain', g.explain));
    for (const id of g.setups) section.append(setupPanel(id));
    const items = data.items.filter(q => q.group === g.id);
    if (items.length && !g.perQuestion) {
      const decide = el('fieldset', 'group-decision');
      decide.append(el('legend', '', items.length > 1 ? 'Decision for all ' + items.length + ' questions below' : 'Decision for this question'));
      decide.append(radios('group-' + g.id, g.options.map(o => [o, g.labels[o] || DECISIONS[o]]), state.groups[g.id].decision,
        v => { state.groups[g.id].decision = v; changed(); }, g.suggested));
      const note = el('details', 'reveal');
      note.append(el('summary', '', 'Comment on this group'), area(state.groups[g.id].comment, 2, v => { state.groups[g.id].comment = v; save(); }));
      note.open = Boolean(state.groups[g.id].comment);
      decide.append(note);
      section.append(decide);
    }
    let lastScenario = null;
    for (const q of items) {
      if (q.scenario_id !== lastScenario) {
        lastScenario = q.scenario_id;
        const s = scenarioById.get(q.scenario_id);
        const block = el('div', 'scenario-block'); block.append(scenarioLabel(s));
        if (!g.setups.includes(s.id)) block.append(el('p', 'setup-text pre', s.preamble));
        section.append(block);
      }
      section.append(questionBlock(q, g));
    }
    refreshers.push(() => {
      const total = items.length + g.setups.length;
      const done = items.filter(questionDone).length + g.setups.filter(setupDone).length;
      count.textContent = done + ' of ' + total + ' settled';
      count.classList.toggle('all', done === total);
      section.hidden = $('only-open').checked && done === total;
      const link = $('toc-' + g.id);
      if (link) link.dataset.done = String(done === total);
    });
    return section;
  }

  function inputRow(s) {
    const row = el('div', 'input-row');
    row.append(el('span', 'sid', s.id), el('span', 'grow', s.title + (s.from_test ? ' · from your ' + s.from_test : '')));
    if (s.needs_input) row.append(el('span', 'need', 'We don’t have your input'));
    const button = el('label', 'file-button', 'Attach input files');
    const file = el('input'); file.type = 'file'; file.multiple = true; file.id = 'attach-' + s.id;
    file.accept = '.pqi,.in,.txt,.phr,.pqo,.out,text/plain';
    button.append(file);
    const list = el('div', 'files');
    function draw() {
      list.replaceChildren(...state.inputs[s.id].map((f, i) => {
        const chip = el('span', 'chip', f.filename);
        const x = el('button', '', '×'); x.type = 'button'; x.setAttribute('aria-label', 'Remove ' + f.filename);
        x.addEventListener('click', () => { state.inputs[s.id].splice(i, 1); save(); draw(); });
        chip.append(x); return chip;
      }));
    }
    file.addEventListener('change', async () => {
      for (const f of file.files) {
        if (f.size > MAX_FILE) { status(f.name + ' is larger than 1 MB and was not attached.'); continue; }
        state.inputs[s.id].push({filename: f.name, text: await f.text()});
      }
      file.value = ''; save(); draw();
    });
    row.append(button, list); draw();
    return row;
  }

  function progress() {
    const q = data.items.filter(questionDone).length, s = setupIds.filter(setupDone).length;
    const total = data.items.length + setupIds.length;
    $('progress-count').textContent = (q + s) + ' of ' + total + ' settled';
    $('progress-bar').style.width = (100 * (q + s) / total) + '%';
  }

  function render() {
    refreshers.length = 0;
    $('reviewer').value = state.reviewer;
    $('toc').replaceChildren(...data.groups.map(g => {
      const li = el('li'); const a = el('a', '', g.title); a.href = '#group-' + g.id; a.id = 'toc-' + g.id;
      const n = g.ids.length + g.setups.length;
      li.append(a, el('span', 'muted', ' · ' + (g.ids.length ? g.ids.length + ' question' + (g.ids.length > 1 ? 's' : '') : '') +
        (g.ids.length && g.setups.length ? ' + ' : '') + (g.setups.length ? g.setups.length + ' setup' + (g.setups.length > 1 ? 's' : '') : '')));
      return n ? li : null;
    }).filter(Boolean));
    $('groups').replaceChildren(...data.groups.map(groupCard));
    $('input-rows').replaceChildren(...data.scenarios.map(inputRow));
    refreshAll(); status();
  }

  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) state = fromRecord(JSON.parse(saved));
  } catch (_) {
    storageOK = false;
  }
  $('reviewer').addEventListener('input', () => { state.reviewer = $('reviewer').value; save(); });
  $('only-open').addEventListener('change', refreshAll);
  $('download').addEventListener('click', () => {
    const rec = record();
    const who = (rec.reviewer || 'author').replace(/[^A-Za-z0-9_-]/g, '_');
    const name = 'phreeqc_recheck_' + who + '_' + rec.exported_at.replace(/[:.]/g, '-') + '.json';
    const url = URL.createObjectURL(new Blob([JSON.stringify(rec, null, 2)], {type: 'application/json'}));
    const link = el('a'); link.href = url; link.download = name;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    status('Downloaded ' + name + ' (' + (rec.settled.questions + rec.settled.setups) + ' of ' +
      (rec.settled.of_questions + rec.settled.of_setups) + ' settled). Send this file back; you can download again any time.');
  });
  $('restore').addEventListener('change', async () => {
    const f = $('restore').files[0];
    if (!f) return;
    try {
      if (f.size > 20 * 1024 * 1024) throw new Error('File is too large.');
      const next = fromRecord(JSON.parse(await f.text()));
      if (!confirm('Replace the answers on this page with the ones in ' + f.name + '?')) return;
      state = next; save(); render(); status('Restored your review from ' + f.name + '.');
    } catch (error) { status('Could not restore: ' + error.message + ' Nothing was changed.'); }
    finally { $('restore').value = ''; }
  });
  render();
})();
