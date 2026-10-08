(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('questionnaire-data').textContent);
  const $ = id => document.getElementById(id);
  const items = data.scenarios.flatMap(s => s.items);
  const ids = items.map(q => q.id);
  const N = ids.length, NS = data.scenarios.length;
  const storageKey = 'phreeqc-human:' + data.meta.selection_sha256;
  const choices = ['A', 'B', 'C', 'D', 'E'];
  const reasons = {
    missing_information: 'Missing information needed to answer',
    no_correct_option: 'None of the options is correct',
    multiple_correct: 'More than one option is correct',
    unclear_wording: 'Wording is unclear or ambiguous',
    other: 'Other (explain in the note)'
  };
  const reuseValues = ['not_recorded', 'shared_run', 'multiple_runs', 'no_run'];
  let storageOK = true;
  let state = fresh();

  function fresh() {
    return {
      schema_version: 2, study_id: data.meta.study_id,
      selection_sha256: data.meta.selection_sha256,
      source_sha256: data.meta.source_sha256,
      database_sha256: data.meta.database_sha256,
      participant_code: '', experience: '', current_scenario: 0,
      created_at: new Date().toISOString(), started_at: null,
      last_saved_at: null, finalized_at: null, locked: false,
      answers: Object.fromEntries(ids.map(id => [id, {choice: null, note: '', problem_reported: false, problem_reason: '', updated_at: null}])),
      simulations: Object.fromEntries(data.scenarios.map(s => [s.id, {reuse: 'not_recorded', notes: ''}]))
    };
  }

  function cleanText(value, max, label) {
    if (typeof value !== 'string' || value.length > max) throw new Error('Invalid ' + label + '.');
    return value;
  }
  function cleanDate(value) {
    if (value === null || value === undefined) return null;
    if (typeof value !== 'string' || !Number.isFinite(Date.parse(value))) throw new Error('Invalid timestamp.');
    return value;
  }
  function normalize(input) {
    if (!input || input.schema_version !== 2 || input.study_id !== data.meta.study_id ||
        input.selection_sha256 !== data.meta.selection_sha256 ||
        input.source_sha256 !== data.meta.source_sha256 ||
        input.database_sha256 !== data.meta.database_sha256) {
      throw new Error('This file belongs to a different questionnaire or dataset version.');
    }
    const result = fresh();
    result.participant_code = cleanText(input.participant_code, 64, 'participant code');
    result.experience = cleanText(input.experience, 240, 'experience');
    result.created_at = cleanDate(input.created_at) || result.created_at;
    result.started_at = cleanDate(input.started_at);
    result.last_saved_at = cleanDate(input.last_saved_at);
    result.finalized_at = cleanDate(input.finalized_at);
    result.current_scenario = Number.isInteger(input.current_scenario) &&
      input.current_scenario >= 0 && input.current_scenario < data.scenarios.length ? input.current_scenario : 0;
    result.locked = input.kind === 'final' || input.locked === true;
    let records;
    if (Array.isArray(input.responses)) {
      if (input.responses.length !== ids.length) throw new Error('The response file must contain all ' + N + ' question records.');
      const unique = new Set(input.responses.map(r => r.question_id));
      if (unique.size !== ids.length || ids.some(id => !unique.has(id))) throw new Error('Question IDs do not match this workbook.');
      records = Object.fromEntries(input.responses.map(r => [r.question_id, r]));
    } else {
      records = input.answers;
      if (!records || Object.keys(records).length !== ids.length) throw new Error('Incomplete saved state.');
    }
    for (const id of ids) {
      const r = records[id];
      if (!r || (r.choice !== null && !choices.includes(r.choice))) throw new Error('Invalid answer for Q' + id + '.');
      if (Array.isArray(input.responses) && r.scenario_id !== data.scenarios.find(s => s.items.some(q => q.id === id)).id) {
        throw new Error('Scenario assignment does not match Q' + id + '.');
      }
      if (typeof r.problem_reported !== 'boolean') throw new Error('Invalid problem report for Q' + id + '.');
      if (r.problem_reason !== '' && !Object.hasOwn(reasons, r.problem_reason)) throw new Error('Invalid problem reason for Q' + id + '.');
      result.answers[id] = {
        choice: r.choice, note: cleanText(r.note, 4000, 'question note'),
        problem_reported: r.problem_reported, problem_reason: r.problem_reported ? r.problem_reason : '',
        updated_at: cleanDate(r.updated_at)
      };
    }
    for (const s of data.scenarios) {
      const r = input.simulations && input.simulations[s.id];
      if (!r || !reuseValues.includes(r.reuse)) throw new Error('Invalid simulation record.');
      result.simulations[s.id] = {reuse: r.reuse, notes: cleanText(r.notes, 4000, 'simulation note')};
    }
    if (result.locked && (!result.participant_code.trim() || ids.some(id => !result.answers[id].choice) || !result.finalized_at)) {
      throw new Error('A final response must have a participant code, ' + N + ' answers, and a finalization timestamp.');
    }
    return result;
  }

  let saved = null;
  try {
    saved = localStorage.getItem(storageKey);
    if (saved) state = normalize(JSON.parse(saved));
  } catch (error) {
    // Keep the unreadable copy so the first save cannot overwrite it.
    try { if (saved) localStorage.setItem(storageKey + ':unreadable:' + Date.now(), saved); } catch (_) {}
    storageOK = false;
    $('message').textContent = 'Saved progress could not be loaded: ' + error.message + ' Restore a backup if available; download backups while working.';
  }

  function save() {
    state.last_saved_at = new Date().toISOString();
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
      storageOK = true;
    } catch (_) { storageOK = false; }
    updateSaveStatus();
  }
  function updateSaveStatus() {
    const element = $('save-status');
    element.classList.toggle('error', !storageOK);
    if (!storageOK) element.textContent = 'Browser storage unavailable — download a backup before closing.';
    else if (state.locked) element.textContent = 'Finalized · editing locked';
    else if (state.last_saved_at) element.textContent = 'Saved in this browser · ' + new Date(state.last_saved_at).toLocaleTimeString();
    else element.textContent = 'Ready · progress will save in this browser';
  }
  function setMessage(text) { $('message').textContent = text; }
  function counts(scenario) {
    const selected = scenario ? scenario.items : items;
    return {
      answered: selected.filter(q => state.answers[q.id].choice !== null).length,
      unsure: selected.filter(q => state.answers[q.id].choice === 'E').length,
      problems: selected.filter(q => state.answers[q.id].problem_reported).length
    };
  }
  function updateProgress() {
    const total = counts();
    $('total-progress').textContent = total.answered + ' / ' + N + ' answered';
    $('idk-progress').textContent = total.unsure + ' unsure · ' + total.problems + ' reported';
    $('progress').value = total.answered;
    const s = data.scenarios[state.current_scenario];
    $('scenario-progress').textContent = counts(s).answered + ' / ' + s.items.length + ' answered';
    data.scenarios.forEach((s, i) => {
      const button = $('nav-' + s.id);
      button.setAttribute('aria-current', String(i === state.current_scenario));
      button.querySelector('.nav-count').textContent = counts(s).answered + '/' + s.items.length;
      button.setAttribute('aria-label', s.id + ': ' + s.title + ', ' + counts(s).answered + ' of ' + s.items.length + ' answered');
    });
    items.forEach(q => {
      const status = $('status-' + q.id);
      if (status) status.textContent = state.answers[q.id].choice === null ? 'Not answered' :
        state.answers[q.id].choice === 'E' ? 'E · I don’t know' : 'Selected ' + state.answers[q.id].choice;
    });
  }
  function answerChanged(id, field, value) {
    if (state.locked) return;
    state.answers[id][field] = value;
    state.answers[id].updated_at = new Date().toISOString();
    if (!state.started_at && field === 'choice' && value !== null) state.started_at = state.answers[id].updated_at;
    save();
    updateProgress();
  }
  function element(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function questionCard(q, scenario) {
    const card = element('fieldset', 'question');
    card.id = 'question-' + q.id;
    const top = element('div', 'question-top');
    top.append(element('span', '', 'Question ' + (ids.indexOf(q.id) + 1) + ' of ' + N + ' · source Q' + q.id));
    const status = element('span'); status.id = 'status-' + q.id; top.append(status);
    card.append(top);
    card.append(element('legend', 'preserve', q.target));
    const content = element('div', 'question-content');
    if (q.preamble !== scenario.preamble) {
      const variant = element('details', 'source-variant');
      variant.open = true;
      variant.append(element('summary', '', 'Original preamble wording for this question'));
      variant.append(element('p', 'preserve', q.preamble));
      content.append(variant);
    }
    const options = element('div', 'options');
    for (const code of choices) {
      const label = element('label', 'option' + (code === 'E' ? ' idk' : ''));
      const radio = document.createElement('input');
      radio.type = 'radio'; radio.name = 'q-' + q.id; radio.value = code;
      radio.id = 'q-' + q.id + '-' + code;
      radio.checked = state.answers[q.id].choice === code;
      radio.disabled = state.locked;
      radio.addEventListener('change', () => answerChanged(q.id, 'choice', code));
      label.append(radio, element('span', 'letter', code),
        element('span', 'preserve', code === 'E' ? 'I don’t know' : q.options[code]));
      options.append(label);
    }
    content.append(options);
    const bottom = element('div', 'question-bottom');
    const flag = element('label', 'flag');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox'; checkbox.id = 'problem-' + q.id;
    checkbox.checked = state.answers[q.id].problem_reported; checkbox.disabled = state.locked;
    flag.append(checkbox, element('span', '', 'Report a problem with this question'));
    const reasonBox = element('div', 'problem');
    reasonBox.hidden = !state.answers[q.id].problem_reported;
    const reasonLabel = element('label', '', 'What is wrong with it?');
    const reason = document.createElement('select');
    reason.id = 'reason-' + q.id; reason.disabled = state.locked;
    reason.append(new Option('Choose a reason…', ''));
    for (const [value, text] of Object.entries(reasons)) reason.append(new Option(text, value));
    reason.value = state.answers[q.id].problem_reason;
    reason.addEventListener('change', () => answerChanged(q.id, 'problem_reason', reason.value));
    reasonLabel.append(reason);
    reasonBox.append(reasonLabel, element('p', 'hint', 'Still choose your best answer above, or E if you cannot tell. Reports go to the question author and do not change your score.'));
    const clear = element('button', 'clear-choice', 'Clear choice');
    clear.type = 'button'; clear.disabled = state.locked;
    clear.addEventListener('click', () => {
      answerChanged(q.id, 'choice', null);
      card.querySelectorAll('input[type=radio]').forEach(r => { r.checked = false; });
    });
    bottom.append(flag, clear); content.append(bottom, reasonBox);
    const details = element('details');
    details.open = Boolean(state.answers[q.id].note) || state.answers[q.id].problem_reported;
    details.append(element('summary', '', 'Note (optional)'));
    checkbox.addEventListener('change', () => {
      answerChanged(q.id, 'problem_reported', checkbox.checked);
      if (!checkbox.checked) { answerChanged(q.id, 'problem_reason', ''); reason.value = ''; }
      reasonBox.hidden = !checkbox.checked;
      if (checkbox.checked) { details.open = true; reason.focus(); }
    });
    const note = document.createElement('textarea');
    note.rows = 2; note.maxLength = 4000; note.value = state.answers[q.id].note; note.disabled = state.locked;
    note.id = 'note-' + q.id; note.setAttribute('aria-label', 'Note for Q' + q.id);
    note.placeholder = 'Describe the problem, or your reason for uncertainty, if useful.';
    note.addEventListener('input', () => answerChanged(q.id, 'note', note.value));
    details.append(note); content.append(details); card.append(content);
    return card;
  }

  function render() {
    document.body.classList.toggle('locked', state.locked);
    $('participant').value = state.participant_code;
    $('experience').value = state.experience;
    $('participant').disabled = state.locked;
    $('experience').disabled = state.locked;
    $('restore').disabled = state.locked;
    $('finish').textContent = state.locked ? 'Download final JSON' : 'Finish & export';
    const scenario = data.scenarios[state.current_scenario];
    $('scenario-position').textContent = scenario.id + ' · SCENARIO ' + (state.current_scenario + 1) + ' OF ' + NS;
    $('scenario-title').textContent = scenario.title;
    $('preamble').textContent = scenario.preamble;
    $('questions').replaceChildren(...scenario.items.map(q => questionCard(q, scenario)));
    $('reuse').value = state.simulations[scenario.id].reuse;
    $('simulation-notes').value = state.simulations[scenario.id].notes;
    $('reuse').disabled = state.locked;
    $('simulation-notes').disabled = state.locked;
    $('previous').disabled = state.current_scenario === 0;
    $('next').disabled = state.current_scenario === data.scenarios.length - 1;
    updateProgress(); updateSaveStatus();
  }
  function navigate(index) {
    state.current_scenario = index;
    save(); render();
    $('main').scrollIntoView({behavior: 'auto', block: 'start'});
  }
  data.scenarios.forEach((s, i) => {
    const button = element('button', 'scenario-link');
    button.type = 'button'; button.id = 'nav-' + s.id;
    button.append(element('span', '', s.id), element('span', 'nav-title', s.title), element('span', 'nav-count', '0/' + s.items.length));
    button.addEventListener('click', () => navigate(i));
    $('scenario-list').append(button);
  });
  $('progress').max = N;
  $('question-count').textContent = N + ' questions · ' + NS + ' scenarios';
  $('version-info').textContent = 'Workbook ' + data.meta.study_id + ' · Selection ' + data.meta.selection_sha256.slice(0, 12) +
    ' · Frozen database SHA-256 ' + data.meta.database_sha256.slice(0, 12) + '…';
  for (const [id, key] of [['participant', 'participant_code'], ['experience', 'experience']]) {
    $(id).addEventListener('input', () => { if (!state.locked) { state[key] = $(id).value; save(); } });
  }
  $('reuse').addEventListener('change', () => {
    if (!state.locked) { state.simulations[data.scenarios[state.current_scenario].id].reuse = $('reuse').value; save(); }
  });
  $('simulation-notes').addEventListener('input', () => {
    if (!state.locked) { state.simulations[data.scenarios[state.current_scenario].id].notes = $('simulation-notes').value; save(); }
  });
  $('previous').addEventListener('click', () => navigate(Math.max(0, state.current_scenario - 1)));
  $('next').addEventListener('click', () => navigate(Math.min(NS - 1, state.current_scenario + 1)));

  function record() {
    return {
      schema_version: 2, kind: state.locked ? 'final' : 'draft',
      study_id: data.meta.study_id, selection_sha256: data.meta.selection_sha256,
      source_sha256: data.meta.source_sha256, database_sha256: data.meta.database_sha256,
      participant_code: state.participant_code.trim(), experience: state.experience,
      created_at: state.created_at, started_at: state.started_at,
      last_saved_at: state.last_saved_at, finalized_at: state.finalized_at,
      exported_at: new Date().toISOString(), current_scenario: state.current_scenario,
      response_count: counts().answered, idk_count: counts().unsure, problem_count: counts().problems,
      protocol: {format: 'MCQ', unknown_choice: 'E', scenario_reuse_allowed: true, feedback_provided: false, problem_reasons: Object.keys(reasons)},
      responses: data.scenarios.flatMap(s => s.items.map(q => ({
        scenario_id: s.id, question_id: q.id, ...state.answers[q.id]
      }))),
      simulations: state.simulations
    };
  }
  function download() {
    const payload = record();
    const code = (payload.participant_code || 'anonymous').replace(/[^A-Za-z0-9_-]/g, '_');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = 'phreeqc_' + code + '_' + payload.kind + '_' + stamp + '.json';
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = filename;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    setMessage((state.locked ? 'Final response' : 'Backup') + ' download requested. Check your downloads and send the final file to the organizer; this page does not upload it.');
  }
  $('backup').addEventListener('click', download);
  $('finish').addEventListener('click', () => {
    if (state.locked) { download(); return; }
    if (!state.participant_code.trim()) { setMessage('Enter your participant code before finalizing.'); $('participant').focus(); return; }
    const total = counts();
    const missingReason = items.find(q => state.answers[q.id].problem_reported && !state.answers[q.id].problem_reason);
    if (missingReason) {
      setMessage('Q' + missingReason.id + ' has a problem report without a reason. Choose a reason, or switch the report off.');
      navigate(data.scenarios.findIndex(s => s.items.includes(missingReason)));
      $('reason-' + missingReason.id).focus(); return;
    }
    if (total.answered !== ids.length) {
      setMessage('You have ' + (ids.length - total.answered) + ' unanswered questions. Choose E for “I don’t know” rather than leaving a blank.');
      const i = data.scenarios.findIndex(s => s.items.some(q => !state.answers[q.id].choice));
      navigate(i); return;
    }
    $('finish-summary').textContent = N + ' of ' + N + ' answered · ' + total.unsure + ' marked E (I don’t know) · ' + total.problems + ' problem reports. Participant: ' + state.participant_code.trim() + '.';
    $('finish-dialog').showModal();
  });
  $('cancel-finish').addEventListener('click', () => $('finish-dialog').close());
  $('confirm-finish').addEventListener('click', () => {
    if (state.locked || counts().answered !== N || !state.participant_code.trim()) return;
    state.locked = true; state.finalized_at = new Date().toISOString();
    save(); $('finish-dialog').close(); render(); download();
  });
  $('restore').addEventListener('change', async () => {
    const file = $('restore').files[0];
    if (!file || state.locked) return;
    try {
      if (file.size > 2 * 1024 * 1024) throw new Error('File is too large.');
      const candidate = normalize(JSON.parse(await file.text()));
      if (!confirm('Restore responses for ' + (candidate.participant_code || '(no code)') +
          '? This replaces the progress currently shown. Download a backup first if needed.')) return;
      state = candidate; save(); render();
      setMessage('Response file restored. ' + (state.locked ? 'This final record is read-only.' : 'You may continue answering.'));
    } catch (error) { setMessage('Restore failed: ' + error.message + ' Current progress was not changed.'); }
    finally { $('restore').value = ''; }
  });
  render();
})();
