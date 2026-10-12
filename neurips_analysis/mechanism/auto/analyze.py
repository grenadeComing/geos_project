#!/usr/bin/env python3
"""
Failure-stage decomposition of PHREEQC agent trajectories (read-only on the repo).

For every (run, verified item) trajectory we assign one stage:
  NO_COMMIT      no gradable final answer (results.jsonl prediction is null)
  CORRECT        final answer correct
  BUILD          wrong; the keyed (gold) value never appears in ANY PHREEQC output the
                 agent's own inputs produced (all executions, re-executed offline)
  NOT_SEEN       wrong; gold value was produced by some execution but never appeared in
                 a tool result the agent received (inline content / read_file / grep_file)
  SEEN_NOT_USED  wrong; gold value appeared in a tool result the agent received
  NONNUMERIC     wrong; item has non-numeric (categorical) or count-type options, not decomposed

"Produced" corpus = everything the agent received + the final workspace output files
+ an offline re-execution (same PHREEQC binary + database) of every execute_phreeqc call,
reconstructed from the write_file events that preceded it.  This matters for the
retrieval protocols, where earlier executions' outputs are never returned inline.

Run:  /usr/bin/python3 analyze.py [--workers 10] [--no-reexec]
Outputs (next to this file): labels.jsonl, summary.md, items.json, reexec_cache/, ref/
"""
import os, re, sys, json, glob, hashlib, subprocess, shutil, argparse, statistics, time
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN, ROUND_DOWN, InvalidOperation
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================ configuration ==================================
REPO = '/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github'
RC = os.path.join(REPO, 'result', 'custom')
SCRATCH = os.path.dirname(os.path.abspath(__file__))
PHREEQC_BIN = '/Users/kezhang/Desktop/projects/geos_PHREEQC/phreeqc-3.8.6-17100/src/phreeqc'
PHREEQC_DB = os.path.join(REPO, 'archive_acm_2026', 'all_materials', 'phreeqc.dat')
DATASETS = os.path.join(REPO, 'datasets')
VERIF = os.path.join(REPO, 'human_evaluation', 'verification.json')
UNION45_FILE = os.path.join(REPO, 'neurips_analysis', 'baseline_snapshot', 'cap_affected_union45_ids.txt')
PHREEQC_TIMEOUT = 200          # same wall-clock cap as the harness
MATCHER_VERSION = 4            # bump to invalidate cached re-execution matches

# TOC: three reruns per model + the single uncapped "tocfix" run spliced in for union45 items
TOC = OrderedDict([
    ('GPT-5.1', (['neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt51',
                  'neurips200_clean_rerun_20260503_alltoc_run2_toc_gpt51',
                  'neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt51'],
                 'neurips200_tocfix_uncapped_capaffected45_toc_gpt51')),
    ('GPT-5.2', (['neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt52',
                  'neurips200_clean_rerun_20260503_alltoc_run2_redo_gpt54_gpt52_toc_gpt52',
                  'neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt52'],
                 'neurips200_tocfix_uncapped_capaffected45_toc_gpt52')),
    ('GPT-5.4', (['neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt_5_4_',
                  'neurips200_clean_rerun_20260503_alltoc_run2_redo_gpt54_gpt52_toc_gpt_5_4_',
                  'neurips200_clean_rerun_20260503_safe_redo_run3_toc_gpt_5_4_'],
                 'neurips200_tocfix_uncapped_capaffected45_toc_gpt_5_4_')),
    ('Claude Sonnet 4.6', (['neurips200_clean_rerun_20260504_sonnet_run1_toc_sonnet46',
                            'neurips200_clean_rerun_20260504_sonnet_run2_toc_sonnet46',
                            'neurips200_clean_rerun_20260504_sonnet_run3_toc_sonnet46'],
                           'neurips200_tocfix_uncapped_capaffected45_toc_sonnet46')),
])
TOCFIX_ONLY = OrderedDict([('Claude Opus 4.6', 'neurips200_tocfix_uncapped_capaffected45_toc_opus46')])
GREP = OrderedDict([('Claude Sonnet 4.6', 'neurips200_grep_sonnet46'),
                    ('GPT-5.1', 'neurips200_grep_gpt51'),
                    ('GPT-5.2', 'neurips200_grep_gpt52')])
RAW = OrderedDict([('GPT-5.1', 'rebuttal_mcq200_raw100k_rerun_gpt51_20260723'),
                   ('GPT-5.2', 'rebuttal_mcq200_raw100k_rerun_gpt52_20260723'),
                   ('Gemini 3 Flash', 'neurips200_clean_agent100k_raw100k_gflash3')])
TOC_CAPPED = OrderedDict([('Gemini 3 Flash', 'neurips200_clean_agent100k_toc_gflash3')])
HYBRID_ROOT = 'hybrid158_verified'     # every <HYBRID_ROOT>/<run>/summary.json that exists is included
REPORT_ROOT = 'report158_verified'     # October 2026: *_tocreport_run* (TOC + read-the-report instruction), *_toc_run* (same-day TOC control)
HYBRID_MODEL = {'gpt-5.1': 'GPT-5.1', 'gpt-5.2': 'GPT-5.2', 'gpt-5.4': 'GPT-5.4',
                'claude-sonnet-4-6': 'Claude Sonnet 4.6', 'claude-opus-4-6': 'Claude Opus 4.6'}
TIER = {'Claude Opus 4.6': 'top', 'Claude Sonnet 4.6': 'top', 'GPT-5.4': 'top',
        'GPT-5.1': 'mid', 'GPT-5.2': 'mid', 'Gemini 3 Flash': 'flash'}

# ============================ number matching =================================
NUM_RE = re.compile(r'(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])')
OPT_RE = re.compile(r'^\s*([ABCD])\.\s*(.*?)\s*$')
SC_COL_RE = re.compile(r'(^|[^a-z])sc([^a-z]|$)|cond', re.I)
ABS_QTY_RE = re.compile(r'dissolved|precipitat|delta', re.I)


def parse_options(question):
    opts = OrderedDict()
    for line in question.replace('\r', '').split('\n'):
        m = OPT_RE.match(line)
        if m and m.group(1) not in opts:
            opts[m.group(1)] = m.group(2)
    return opts


def to_dec(s):
    s = s.strip()
    if not re.fullmatch(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', s):
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def make_spec(opt_texts, absmode):
    """opt_texts: {letter: text}.  Returns list of option specs (plain tuples, picklable)."""
    spec = []
    for L, t in opt_texts.items():
        d = to_dec(t)
        e = d.as_tuple().exponent
        intctx = ('.' not in t and 'e' not in t.lower())
        o = abs(d) if absmode else d
        u = float(Decimal(1).scaleb(e))
        spec.append((L, t, str(d), e, absmode, intctx, float(o) - u, float(o) + u))
    return spec


def tok_match(tokstr, sp, mode='round', hp=False):
    """Return 'exact' / 'rounded' (mode='round'|'round_strict') or 'trunc' (mode='trunc') if the printed
    token matches option spec sp, else None.  hp=True marks selected-output-style text (agent-defined
    SELECTED_OUTPUT / USER_PUNCH files), where any finer-precision token may be rounded."""
    L, t, dstr, e, absmode, intctx, lo, hi = sp
    try:
        f = float(tokstr)
    except ValueError:
        return None
    if absmode:
        f = abs(f)
    if f < lo or f > hi:
        return None
    tv = Decimal(tokstr)
    o = Decimal(dstr)
    if absmode:
        tv, o = abs(tv), abs(o)
    te = tv.as_tuple().exponent
    diff = e - te                     # >0: token printed with finer precision than the option
    if diff < 0:
        return None
    if tv == o:                        # exact, incl. trailing zeros (2.4960e-03 == 2.496e-03)
        return 'exact' if mode != 'trunc' else None
    sci = 'e' in tokstr.lower()
    # >=6 significant digits in fixed notation only occurs in selected-output (%g) printing
    sig6 = (not sci) and len(re.sub(r'[^0-9]', '', tokstr).lstrip('0')) >= 6
    if not (diff >= 2 or (diff >= 1 and (sci or intctx or sig6 or hp))):
        return None
    q = Decimal(1).scaleb(e)
    up = tv.quantize(q, rounding=ROUND_HALF_UP)
    dn = tv.quantize(q, rounding=ROUND_HALF_DOWN)
    half = (up != dn)                  # token sits exactly on a rounding midpoint (e.g. 7.9485e+03)
    if mode == 'round':
        if o in (up, dn):
            return 'half' if half else 'rounded'
        return None
    if mode == 'round_strict':
        return 'rounded' if (not half and up == o) else None
    return 'trunc' if (not half and tv.quantize(q, rounding=ROUND_DOWN) == o) else None


def clean_text(text, kind):
    """kind: 'out' (PHREEQC main output: drop tab-indented input-echo lines),
    'grep' (strip 'NNN: ' prefixes, then same), 'plain' (selected output etc.)."""
    if not text:
        return ''
    if kind == 'grep':
        text = '\n'.join(re.sub(r'^\d+: ', '', ln) for ln in text.split('\n'))
        kind = 'out'
    if kind == 'out':
        # drop the echoed input deck: lines between "Reading input data for simulation N."
        # (and its dashed underline) and the next all-dash line
        keep = []
        state = 0      # 0 normal, 1 saw header (skip underline), 2 inside echo
        for ln in text.split('\n'):
            st = ln.strip()
            if 'Reading input data for simulation' in ln:
                state = 1
                keep.append(ln)
                continue
            if state == 1:
                if re.match(r'^-{3,}$', st):
                    state = 2
                keep.append(ln)
                continue
            if state == 2:
                if re.match(r'^-{3,}$', st):
                    state = 0
                    keep.append(ln)
                continue
            keep.append(ln)
        text = '\n'.join(keep)
    return text


def _iter_candidates(text, spec):
    """Yield (token, [specs to test]) honoring the SC-context rule for integer options."""
    plain = [sp for sp in spec if not sp[5]]
    ctx = [sp for sp in spec if sp[5]]
    if plain:
        for m in NUM_RE.finditer(text):
            yield m.group(0), plain
    if ctx:   # integer-valued (specific conductance) options: require SC context
        header_idx = None
        for ln in text.split('\n'):
            low = ln.lower()
            if 'onductance' in low:
                for m in NUM_RE.finditer(ln):
                    yield m.group(0), ctx
                continue
            fields = ln.split()
            if not fields:
                continue
            if all(to_dec(f) is None for f in fields):
                header_idx = ([i for i, f in enumerate(fields) if SC_COL_RE.search(f)], len(fields))
                continue
            if header_idx and header_idx[0] and len(fields) == header_idx[1]:
                for i in header_idx[0]:
                    yield fields[i], ctx


def match_text(text, spec, kinds=None, mode='round', hp=False):
    """Return Counter {letter: count} of option values found in text.  If `kinds` (a Counter)
    is given, also count (letter, 'exact'|'rounded'|'trunc')."""
    found = Counter()
    if not text:
        return found
    for tok, sps in _iter_candidates(text, spec):
        for sp in sps:
            k = tok_match(tok, sp, mode, hp)
            if k:
                found[sp[0]] += 1
                if kinds is not None:
                    kinds[(sp[0], k)] += 1
    return found


def round_trunc_pairs(text, spec, hp=False):
    """Counter {(round_letter, trunc_letter)}: high-precision tokens that ROUND to one option and
    TRUNCATE to a different option (a truncation-instead-of-rounding trap)."""
    out = Counter()
    if not text:
        return out
    for tok, sps in _iter_candidates(text, spec):
        rl = [sp[0] for sp in sps if tok_match(tok, sp, 'round_strict', hp) == 'rounded']
        if not rl:
            continue
        tl = [sp[0] for sp in sps if tok_match(tok, sp, 'trunc', hp)]
        for a in rl:
            for b in tl:
                if a != b:
                    out[(a, b)] += 1
    return out


# ============================ items =========================================

def load_items():
    ver = json.load(open(VERIF))
    ds = [json.loads(l) for l in open(os.path.join(DATASETS, 'neurips_dataset_200_clean.jsonl'))]
    items = OrderedDict()
    for q in ver['questions']:
        if q['status'] != 'verified':
            continue
        d = ds[q['id'] - 1]
        assert d['id'] == q['id']
        opts = parse_options(d['question'])
        gold = q['gold']
        assert gold == d['answer'], (q['id'], gold, d['answer'])
        qty = q.get('quantity', '')
        decs = {L: to_dec(t) for L, t in opts.items()}
        if len(opts) == 4 and all(v is not None for v in decs.values()):
            kind = 'count' if qty.lower().startswith('number of') else 'numeric'
        else:
            kind = 'categorical'
        it = dict(id=q['id'], scenario=q.get('scenario'), quantity=qty, gold=gold,
                  gold_value=q.get('gold_value'), options=dict(opts), kind=kind,
                  question=d['question'])
        if kind == 'numeric':
            it['absmode'] = bool(ABS_QTY_RE.search(qty))
            it['spec'] = make_spec(opts, it['absmode'])
            it['intctx'] = any(sp[5] for sp in it['spec'])
        items[q['id']] = it
    return items, ver


# ============================ trajectories ===================================

def norm_path(p):
    if p is None:
        return ''
    p = str(p).strip().replace('\\', '/')
    if not p:
        return ''
    m = re.search(r'dataset_q_\d+/(.*)$', p)
    if m:
        p = m.group(1)
    elif p.startswith('/'):
        p = os.path.basename(p)
    p = os.path.normpath(p)
    return '' if p == '.' else p


def pick_log(ws):
    logs = sorted(glob.glob(os.path.join(ws, 'logs', 'chat_*.jsonl')))
    if not logs:
        return None, 0, 'no_log'
    complete = []
    for lp in logs:
        try:
            with open(lp) as f:
                if any('"run_end"' in ln for ln in f):
                    complete.append(lp)
        except OSError:
            pass
    if complete:
        return complete[-1], len(logs), 'latest_complete'
    return logs[-1], len(logs), 'latest_incomplete'


def as_dict(r):
    if isinstance(r, dict):
        return r
    if isinstance(r, str):
        try:
            return json.loads(r)
        except Exception:
            import ast
            try:
                return ast.literal_eval(r)
            except Exception:
                return {'_raw': r}
    return {}


def parse_log(lp):
    ev = []
    with open(lp) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ev.append(json.loads(ln))
            except Exception:
                pass
    return ev


_DMAP = {}
_RES = {}


def run_dataset_map(run_dir):
    """Map workspace number -> original item id using the run's dataset file (row order)."""
    if run_dir in _DMAP:
        return _DMAP[run_dir]
    sj = os.path.join(run_dir, 'summary.json')
    dsname = 'neurips_dataset_200_clean.jsonl'
    if os.path.exists(sj):
        dsname = json.load(open(sj)).get('dataset') or dsname
    rows = [json.loads(l) for l in open(os.path.join(DATASETS, dsname))]
    _DMAP[run_dir] = {k + 1: (int(r['id']), r['question']) for k, r in enumerate(rows)}
    return _DMAP[run_dir]


def load_results(run_dir):
    if run_dir in _RES:
        return _RES[run_dir]
    out = {}
    for l in open(os.path.join(run_dir, 'results.jsonl')):
        r = json.loads(l)
        out[int(r['index'])] = r
    _RES[run_dir] = out
    return out


TRAJ_CACHE = {}


def build_traj(run_dir, item_id, items):
    """Parse one trajectory (cached per run_dir,item)."""
    key = (run_dir, item_id)
    if key in TRAJ_CACHE:
        return TRAJ_CACHE[key]
    dmap = run_dataset_map(run_dir)
    ws_no = [k for k, (i, _) in dmap.items() if i == item_id]
    if not ws_no:
        TRAJ_CACHE[key] = None
        return None
    ws = os.path.join(run_dir, 'work_space', 'dataset_q_%05d' % ws_no[0])
    res = load_results(run_dir).get(item_id)
    lp, nlogs, logpick = pick_log(ws)
    t = dict(run_dir=run_dir, ws=ws, item_id=item_id, n_logs=nlogs, log_pick=logpick,
             log=lp, result_row=res)
    if lp is None:
        t['events'] = []
    else:
        t['events'] = parse_log(lp)
    # verify mapping through question text in the first user event
    users = [e for e in t['events'] if e.get('action') == 'user']
    qtext = dmap[ws_no[0]][1].replace('\r', '').strip()
    t['map_ok'] = bool(users) and qtext[:400] in users[0].get('content', '').replace('\r', '')
    TRAJ_CACHE[key] = t
    return t


def analyse_events(t):
    """Walk the tool events: collect received output pieces, exec jobs, process metrics."""
    ev = t['events']
    files = {}
    written = set()
    out_paths = {'result.out'}
    pieces = []          # (step, channel, text) received output-derived text
    execs = []           # (step, job_key, files_snapshot, input_path)
    m = Counter()
    exec_results = []
    rr_reads = []        # (step, start, end) result.out ranged reads (after which exec)
    steps = 0
    last_exec_step = None
    for e in ev:
        a = e.get('action')
        if a == 'assistant':
            steps += 1
            continue
        if a != 'tool':
            continue
        tool = e.get('tool')
        args = e.get('args') or {}
        if not isinstance(args, dict):
            args = {}
        r = as_dict(e.get('result'))
        step = e.get('step')
        m['n_tool'] += 1
        m['n_' + str(tool)] += 1
        if tool == 'write_file':
            p = norm_path(args.get('path'))
            if r.get('ok') and p:
                files[p] = args.get('new_content', '') or ''
                written.add(p)
                cu = files[p].upper()
                if 'SELECTED_OUTPUT' in cu or 'USER_PUNCH' in cu:
                    m['sel_defined'] += 1
                if re.search(r'(?im)^\s*PRINT\b[\s\S]{0,400}?-?reset\s+(false|f\b)', files[p]):
                    m['print_reset_false'] += 1
        elif tool == 'execute_phreeqc':
            inp = norm_path(args.get('input_path'))
            snap = dict(files)
            k = hashlib.sha1(json.dumps([sorted(snap.items()), inp]).encode()).hexdigest()
            execs.append((step, k, snap, inp))
            last_exec_step = step
            d = os.path.dirname(inp)
            for f in (r.get('output_files') or []):
                out_paths.add(os.path.normpath(os.path.join(d, f)) if d else f)
            if r.get('ok'):
                m['n_exec_ok'] += 1
            sizes = r.get('file_sizes') or {}
            rsz = sizes.get('result.out')
            if 'result_out_toc' in r:
                if r.get('result_out_toc'):
                    m['toc_nonempty'] += 1
                elif rsz and rsz > 100000:
                    m['toc_empty_capped'] += 1
            if r.get('result_out_content'):
                pieces.append((step, 'inline', clean_text(r['result_out_content'], 'out')))
                m['inline'] += 1
                if 'TRUNCATED' in r['result_out_content'][:200000] or r.get('warning'):
                    m['inline_truncated'] += 1
            exec_results.append((step, rsz))
        elif tool in ('read_file', 'grep_file'):
            p = norm_path(args.get('path'))
            base = os.path.basename(p)
            is_input = (p in written and p not in out_paths)
            excluded = base in ('database_reference.txt', 'final_answer.txt') or p.startswith('logs')
            is_output = (p in out_paths) or (not is_input and not excluded and p != '')
            if is_input:
                m['n_read_input'] += 1
            if not is_output or excluded:
                continue
            is_main = (base == 'result.out')   # the harness always writes the main output to result.out
            if tool == 'read_file':
                m['n_read_output'] += 1
                if base == 'result.out':
                    m['n_read_resultout'] += 1
                    if 'first' in str(r.get('showing', '')):
                        m['n_read_preview'] += 1
                    sl, el = args.get('start_line'), args.get('end_line')
                    try:
                        rr_reads.append((step, int(sl), int(el), last_exec_step))
                    except (TypeError, ValueError):
                        rr_reads.append((step, None, None, last_exec_step))
                else:
                    m['n_read_other_output'] += 1
                txt = r.get('content') or ''
                if txt:
                    pieces.append((step, 'read_main' if is_main else 'read_sel',
                                   clean_text(txt, 'out' if is_main else 'plain')))
            else:
                m['n_grep_output'] += 1
                if base == 'result.out':
                    m['n_grep_resultout'] += 1
                txt = '\n'.join(x.get('context', '') for x in (r.get('matches') or []) if isinstance(x, dict))
                if txt:
                    pieces.append((step, 'grep' if is_main else 'grep_sel', clean_text(txt, 'grep')))
    run_end = [e for e in ev if e.get('action') == 'run_end']
    t['status'] = run_end[-1].get('status') if run_end else None
    t['steps'] = steps
    t['pieces'] = pieces
    t['execs'] = execs
    t['metrics'] = m
    t['written'] = written
    t['out_paths'] = out_paths
    t['rr_reads'] = rr_reads
    t['last_exec_step'] = last_exec_step


def final_outputs(t):
    """Output files left in the workspace (last execution)."""
    texts = []
    ws = t['ws']
    for p in sorted(glob.glob(os.path.join(ws, '**', '*'), recursive=True)):
        if not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, ws)
        base = os.path.basename(rel)
        if rel.startswith('logs') or base in ('database_reference.txt', 'final_answer.txt', 'phreeqc.log'):
            continue
        if rel in t['written'] and rel not in t['out_paths']:
            continue
        if base == 'result.out' or rel in t['out_paths'] or base.endswith(('.sel', '.out', '.txt', '.csv', '.tsv', '.dat', '.xls')):
            try:
                txt = open(p, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            texts.append((rel, clean_text(txt, 'out' if base == 'result.out' else 'plain')))
    return texts


# ============================ re-execution ====================================

def norm_out_sha(text):
    """sha1 of result.out without the path header lines, the run-time line and dash rules."""
    lines = text.split('\n')
    keep = [l for l in lines if not re.match(r'^\s*(Input file|Output file|Database file):', l)
            and 'End of Run after' not in l and not re.match(r'^-+$', l.strip())]
    return hashlib.sha1('\n'.join(keep).encode('utf-8', 'replace')).hexdigest()


def reexec_worker(key, files, inp, item_specs):
    """Run PHREEQC on a reconstructed workspace; return option-presence per item."""
    tmp = os.path.join(SCRATCH, 'reexec_tmp', key)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    out = dict(key=key, ok=False, rc=None, timeout=False, out_chars=0, sha=None, items={}, mv=MATCHER_VERSION)
    try:
        for p, c in files.items():
            fp = os.path.join(tmp, p)
            if not os.path.abspath(fp).startswith(tmp):
                continue
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, 'w', encoding='utf-8') as f:
                f.write(c)
        fin = os.path.join(tmp, inp)
        if not os.path.exists(fin):
            out['error'] = 'input_missing'
            return out
        wd = os.path.dirname(fin)
        before = set(os.listdir(wd))
        try:
            p = subprocess.run([PHREEQC_BIN, fin, os.path.join(wd, 'result.out'), PHREEQC_DB],
                               cwd=wd, capture_output=True, text=True, timeout=PHREEQC_TIMEOUT)
            out['rc'] = p.returncode
            out['ok'] = p.returncode == 0
        except subprocess.TimeoutExpired:
            out['timeout'] = True
        texts = []
        ro = os.path.join(wd, 'result.out')
        if os.path.exists(ro):
            txt = open(ro, encoding='utf-8', errors='replace').read()
            out['out_chars'] = len(txt)
            out['sha2'] = norm_out_sha(txt)
            texts.append((clean_text(txt, 'out'), False))
        for f in sorted(set(os.listdir(wd)) - before):
            fp = os.path.join(wd, f)
            if f in ('result.out', 'phreeqc.log') or not os.path.isfile(fp):
                continue
            if os.path.relpath(fp, tmp) in files:
                continue
            try:
                texts.append((clean_text(open(fp, encoding='utf-8', errors='replace').read(), 'plain'), True))
            except OSError:
                pass
        for iid, spec in item_specs.items():
            c = Counter()
            for tx, hp in texts:
                c.update(match_text(tx, spec, hp=hp))
            out['items'][str(iid)] = dict(c)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def run_reexec(jobs, item_specs_all, workers):
    cache_dir = os.path.join(SCRATCH, 'reexec_cache')
    os.makedirs(cache_dir, exist_ok=True)
    results = {}
    todo = []
    for k, (files, inp, iids) in jobs.items():
        cp = os.path.join(cache_dir, k + '.json')
        if os.path.exists(cp):
            try:
                c = json.load(open(cp))
                if (all(str(i) in c['items'] for i in iids) and ('sha2' in c or not c.get('out_chars'))
                        and c.get('mv') == MATCHER_VERSION):
                    results[k] = c
                    continue
            except Exception:
                pass
        todo.append(k)
    print('[reexec] %d jobs, %d cached, %d to run' % (len(jobs), len(results), len(todo)), flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for k in todo:
                files, inp, iids = jobs[k]
                futs[ex.submit(reexec_worker, k, files, inp, {i: item_specs_all[i] for i in iids})] = k
            for n, fu in enumerate(as_completed(futs), 1):
                k = futs[fu]
                try:
                    r = fu.result()
                except Exception as e:  # noqa
                    r = dict(key=k, ok=False, error=repr(e), items={})
                results[k] = r
                with open(os.path.join(cache_dir, k + '.json'), 'w') as f:
                    json.dump(r, f)
                if n % 250 == 0:
                    print('[reexec] %d/%d  %.0fs' % (n, len(todo), time.time() - t0), flush=True)
    return results


# ============================ reference outputs ===============================

PQ_KEYWORDS = set(k.upper() for k in """SOLUTION SOLUTION_SPREAD EQUILIBRIUM_PHASES EXCHANGE SURFACE GAS_PHASE KINETICS
RATES REACTION REACTION_TEMPERATURE REACTION_PRESSURE MIX USE SAVE END TITLE TRANSPORT ADVECTION
INCREMENTAL_REACTIONS KNOBS SELECTED_OUTPUT USER_PUNCH USER_PRINT USER_GRAPH PRINT PHASES SOLUTION_SPECIES
SOLUTION_MASTER_SPECIES EXCHANGE_SPECIES SURFACE_SPECIES EXCHANGE_MASTER_SPECIES SURFACE_MASTER_SPECIES
INVERSE_MODELING COPY DELETE RUN_CELLS DUMP SOLID_SOLUTIONS DATABASE INCLUDE$ CALCULATE_VALUES ISOTOPES
ISOTOPE_ALPHAS ISOTOPE_RATIOS NAMED_EXPRESSIONS PITZER SIT LLNL_AQUEOUS_MODEL_PARAMETERS
REACTION_TEMPERATURE_RAW MEAN_GAMMAS""".split())
OUTPUT_KEYWORDS = {'PRINT', 'SELECTED_OUTPUT', 'USER_PUNCH', 'USER_PRINT', 'USER_GRAPH'}


def _kw(line):
    tok = line.strip().split()
    if not tok:
        return None
    k = tok[0].upper()
    if k in PQ_KEYWORDS or k.endswith('_MODIFY') or k.endswith('_RAW'):
        return k
    return None


def strip_output_controls(content):
    """Remove the agent's output-shaping blocks (PRINT / SELECTED_OUTPUT / USER_PUNCH / USER_PRINT /
    USER_GRAPH) and TRANSPORT/ADVECTION print/punch options, keeping its chemistry unchanged, so PHREEQC
    prints its default full report for every step/cell."""
    out = []
    block = None
    for ln in content.split('\n'):
        k = _kw(ln)
        if k:
            block = k
            if k in OUTPUT_KEYWORDS:
                continue
            out.append(ln)
            continue
        if block in OUTPUT_KEYWORDS:
            continue
        if block in ('TRANSPORT', 'ADVECTION') and re.match(r'^\s*-?(print|punch|warnings|dump)', ln, re.I):
            continue
        out.append(ln)
    return '\n'.join(out)


def reference_presence(items, ver):
    """Run each scenario's reference input (human_evaluation/verification_inputs) and count
    which options of each verified numeric item appear in the reference result.out."""
    refdir = os.path.join(SCRATCH, 'ref')
    os.makedirs(refdir, exist_ok=True)
    scen = {s['id']: s for s in ver['scenarios']}
    texts = {}
    for sid, s in scen.items():
        mi = s.get('main_input')
        if not mi:
            continue
        src = os.path.join(REPO, 'human_evaluation', mi)
        d = os.path.join(refdir, sid)
        os.makedirs(d, exist_ok=True)
        outp = os.path.join(d, 'ref.out')
        if not os.path.exists(outp):
            shutil.copy(src, os.path.join(d, os.path.basename(src)))
            subprocess.run([PHREEQC_BIN, os.path.basename(src), 'ref.out', PHREEQC_DB], cwd=d,
                           capture_output=True, text=True, timeout=600)
        if os.path.exists(outp):
            texts[sid] = clean_text(open(outp, encoding='utf-8', errors='replace').read(), 'out')
    ref = {}
    for iid, it in items.items():
        if it['kind'] != 'numeric':
            continue
        tx = texts.get(it['scenario'], '')
        c = match_text(tx, it['spec'])
        ref[iid] = dict(letters=sorted(c), gold_count=c.get(it['gold'], 0), have_ref=bool(tx))
    return ref


# ============================ main assembly ===================================

def collect_specs():
    """List of (model, protocol, run_label, run_dir, src) per item-selection rule."""
    union45 = set(int(x) for x in open(UNION45_FILE).read().split())
    specs = []
    for model, (reruns, fix) in TOC.items():
        for i, rd in enumerate(reruns, 1):
            specs.append(dict(model=model, protocol='TOC', run='r%d' % i, run_dir=os.path.join(RC, rd),
                              fix_dir=os.path.join(RC, fix), splice=True))
    for model, rd in TOCFIX_ONLY.items():
        specs.append(dict(model=model, protocol='TOCfix45', run='fix', run_dir=os.path.join(RC, rd), only_union=True))
    for model, rd in GREP.items():
        specs.append(dict(model=model, protocol='Grep', run='r1', run_dir=os.path.join(RC, rd)))
    for model, rd in RAW.items():
        specs.append(dict(model=model, protocol='Raw100k', run='r1', run_dir=os.path.join(RC, rd)))
    for model, rd in TOC_CAPPED.items():
        specs.append(dict(model=model, protocol='TOC-capped', run='r1', run_dir=os.path.join(RC, rd)))
    hroot = os.path.join(RC, HYBRID_ROOT)
    for sj in sorted(glob.glob(os.path.join(hroot, '*', 'summary.json'))):
        rd = os.path.dirname(sj)
        if 'smoke' in rd:
            continue
        s = json.load(open(sj))
        model = HYBRID_MODEL.get(s.get('model'), s.get('model'))
        specs.append(dict(model=model, protocol='Hybrid', run=os.path.basename(rd), run_dir=rd))
    for sj in sorted(glob.glob(os.path.join(RC, REPORT_ROOT, '*', 'summary.json'))):
        rd = os.path.dirname(sj)
        if 'smoke' in rd:
            continue
        s = json.load(open(sj))
        model = HYBRID_MODEL.get(s.get('model'), s.get('model'))
        b = os.path.basename(rd)
        proto = ('TOC+Report' if '_tocreport_' in b else 'TOC+Round' if '_tocround_' in b else 'Raw100k' if '_raw100k_' in b else 'TOC-Oct')
        specs.append(dict(model=model, protocol=proto, run=os.path.basename(rd), run_dir=rd))
    return specs, union45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--no-reexec', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()
    if args.summary_only:
        summarize_main()
        return

    items, ver = load_items()
    specs, union45 = collect_specs()
    print('[items] verified=%d kinds=%s' % (len(items), dict(Counter(i['kind'] for i in items.values()))))

    # ---- assemble trajectory rows
    rows = []
    for sp in specs:
        for iid, it in items.items():
            src_dir = sp['run_dir']
            src = 'single'
            if sp.get('splice'):
                if iid in union45:
                    src_dir, src = sp['fix_dir'], 'tocfix'
                else:
                    src = 'rerun'
            if sp.get('only_union') and iid not in union45:
                continue
            t = build_traj(src_dir, iid, items)
            if t is None:
                continue          # item not in this run (e.g. hybrid run is 158-only, fine)
            rows.append(dict(spec=sp, item=it, traj=t, src=src))
    print('[rows] %d trajectory rows' % len(rows))

    # ---- parse events once per unique trajectory
    uniq = {}
    for r in rows:
        k = (r['traj']['run_dir'], r['item']['id'])
        if k not in uniq:
            analyse_events(r['traj'])
            uniq[k] = r['traj']
    print('[traj] %d unique trajectories' % len(uniq))

    # ---- re-execution jobs
    jobs = {}
    for (rd, iid), t in uniq.items():
        it = items[iid]
        for (step, k, snap, inp) in t['execs']:
            if k not in jobs:
                jobs[k] = (snap, inp, set())
            if it['kind'] == 'numeric':
                jobs[k][2].add(iid)
    item_specs_all = {iid: it['spec'] for iid, it in items.items() if it['kind'] == 'numeric'}
    reexec = {} if args.no_reexec else run_reexec(jobs, item_specs_all, args.workers)

    ref = reference_presence(items, ver)

    # ---- per unique trajectory: matches
    for (rd, iid), t in uniq.items():
        it = items[iid]
        res = t['result_row'] or {}
        t['pred'] = res.get('prediction')
        t['is_correct'] = res.get('is_correct')
        fa = os.path.join(t['ws'], 'final_answer.txt')
        t['final_answer_txt'] = open(fa).read().strip()[:80] if os.path.exists(fa) else None
        if it['kind'] != 'numeric':
            continue
        spec = it['spec']
        recv = Counter()
        recv_by_channel = defaultdict(Counter)
        first_seen = {}
        kinds = Counter()
        pairs = Counter()
        for (step, ch, txt) in t['pieces']:
            hp = ch in ('read_sel', 'grep_sel')
            c = match_text(txt, spec, kinds, hp=hp)
            pairs.update(round_trunc_pairs(txt, spec, hp=hp))
            recv.update(c)
            recv_by_channel[ch].update(c)
            for L in c:
                first_seen.setdefault(L, step)
        fin = Counter()
        fin_ro = Counter()
        gold_lines = []
        for rel, txt in final_outputs(t):
            c = match_text(txt, spec, hp=os.path.basename(rel) != 'result.out')
            fin.update(c)
            if os.path.basename(rel) == 'result.out':
                fin_ro.update(c)
        # line numbers of gold in the final result.out (for read-range distance)
        ro = os.path.join(t['ws'], 'result.out')
        if os.path.exists(ro) and fin_ro.get(it['gold']):
            gsp = [s for s in spec if s[0] == it['gold']]
            raw = open(ro, encoding='utf-8', errors='replace').read().split('\n')
            cl = clean_text('\n'.join(raw), 'out').split('\n')
            keepset = None
            if len(cl) != len(raw):
                # map kept lines back to original numbers
                keepset = set()
                j = 0
                for k_, ln in enumerate(raw):
                    if j < len(cl) and ln == cl[j]:
                        keepset.add(k_)
                        j += 1
            for ln_no, ln in enumerate(raw, 1):
                if keepset is not None and (ln_no - 1) not in keepset:
                    continue
                if match_text(ln, gsp):
                    gold_lines.append(ln_no)
        rex = Counter()
        rex_fail = 0
        rex_n = 0
        for (step, k, snap, inp) in t['execs']:
            r = reexec.get(k)
            if r is None:
                continue
            rex_n += 1
            if r.get('timeout') or r.get('error'):
                rex_fail += 1
            rex.update(r.get('items', {}).get(str(iid), {}))
        # validate last exec re-run vs final result.out
        val = None
        if t['execs'] and os.path.exists(ro):
            r = reexec.get(t['execs'][-1][1])
            if r and r.get('sha2'):
                val = (r['sha2'] == norm_out_sha(open(ro, encoding='utf-8', errors='replace').read()))
        t['recv'] = recv
        t['recv_kinds'] = kinds
        t['rt_pairs'] = pairs
        t['recv_by_channel'] = recv_by_channel
        t['first_seen'] = first_seen
        t['fin'] = fin
        t['fin_ro'] = fin_ro
        t['rex'] = rex
        t['rex_fail'] = rex_fail
        t['rex_n'] = rex_n
        t['reexec_valid'] = val
        t['gold_lines'] = gold_lines[:50]
        t['n_gold_lines'] = len(gold_lines)

    # ---- default-print re-runs for trajectories whose gold value was never produced:
    # same chemistry, output-shaping blocks removed -> separates chemistry errors from print/probe errors
    djobs = {}
    for (rd, iid), t in uniq.items():
        it = items[iid]
        t['default_letters'] = None
        if it['kind'] != 'numeric' or t.get('is_correct') is True:
            continue
        produced = set(t['recv']) | set(t['fin']) | set(t['rex'])
        if it['gold'] in produced:
            continue
        t['default_keys'] = []
        for (step, k, snap, inp) in t['execs']:
            if inp not in snap:
                continue
            snap2 = dict(snap)
            snap2[inp] = strip_output_controls(snap[inp])
            k2 = hashlib.sha1(json.dumps(['default', sorted(snap2.items()), inp]).encode()).hexdigest()
            t['default_keys'].append(k2)
            if k2 not in djobs:
                djobs[k2] = (snap2, inp, set())
            djobs[k2][2].add(iid)
    dres = {} if args.no_reexec else run_reexec(djobs, item_specs_all, args.workers)
    for (rd, iid), t in uniq.items():
        if t.get('default_keys') is None or 'default_keys' not in t:
            continue
        c = Counter()
        for k2 in t['default_keys']:
            r = dres.get(k2)
            if r:
                c.update(r.get('items', {}).get(str(iid), {}))
        t['default_letters'] = c

    # ---- label rows
    out_rows = []
    for r in rows:
        sp, it, t = r['spec'], r['item'], r['traj']
        gold = it['gold']
        m = t['metrics']
        row = OrderedDict(model=sp['model'], tier=TIER.get(sp['model'], '?'), protocol=sp['protocol'],
                          run=sp['run'], source=r['src'], source_dir=os.path.relpath(t['run_dir'], RC),
                          item_id=it['id'], scenario=it['scenario'], kind=it['kind'],
                          numeric=it['kind'] == 'numeric', in_union45=None,
                          correct=bool(t['is_correct']) if t['is_correct'] is not None else None,
                          pred=t['pred'], gold=gold, final_answer=t['final_answer_txt'],
                          status=t['status'], steps=t['steps'], n_logs=t['n_logs'], log_pick=t['log_pick'],
                          map_ok=t['map_ok'])
        row['n_tool'] = m.get('n_tool', 0)
        row['n_exec'] = m.get('n_execute_phreeqc', 0)
        row['n_read'] = m.get('n_read_file', 0)
        row['n_read_output'] = m.get('n_read_output', 0)
        row['n_read_resultout'] = m.get('n_read_resultout', 0)
        row['n_read_preview'] = m.get('n_read_preview', 0)
        row['n_read_other_output'] = m.get('n_read_other_output', 0)
        row['n_grep'] = m.get('n_grep_file', 0)
        row['n_grep_resultout'] = m.get('n_grep_resultout', 0)
        row['toc_nonempty'] = m.get('toc_nonempty', 0)
        row['toc_empty_capped'] = m.get('toc_empty_capped', 0)
        row['inline'] = m.get('inline', 0)
        row['inline_truncated'] = m.get('inline_truncated', 0)
        row['sel_defined'] = m.get('sel_defined', 0) > 0
        row['print_reset_false'] = m.get('print_reset_false', 0) > 0
        # stage
        if t['is_correct'] is None:
            stage = 'NO_COMMIT'
            fa = (t['final_answer_txt'] or '')
            if t['status'] == 'max_steps_reached' and not fa:
                sub = 'budget'
            elif fa:
                sub = 'declined'      # final answer present but not a letter (None/X/value)
            else:
                sub = 'other'
            row['no_commit_reason'] = sub
        elif t['is_correct']:
            stage = 'CORRECT'
        elif it['kind'] != 'numeric':
            stage = 'NONNUMERIC'
        else:
            stage = None
        notes = []
        if it['kind'] == 'numeric':
            recv, fin, rex = t['recv'], t['fin'], t['rex']
            produced = set(recv) | set(fin) | set(rex)
            produced_norex = set(recv) | set(fin)
            seen = gold in recv
            prod = gold in produced
            row['seen_flag'] = seen
            row['produced_flag'] = prod
            row['produced_flag_without_reexec'] = gold in produced_norex
            row['gold_in_final_resultout'] = gold in t['fin_ro']
            row['n_options_present'] = len(produced & set('ABCD'))
            row['options_present'] = ''.join(sorted(produced))
            row['n_options_seen'] = len(set(recv))
            row['options_seen'] = ''.join(sorted(recv))
            row['distractor_present'] = bool(produced - {gold})
            row['distractor_seen'] = bool(set(recv) - {gold})
            gk = t['recv_kinds']
            row['gold_seen_kind'] = ('exact' if gk.get((gold, 'exact')) else
                                     ('rounded_only' if gk.get((gold, 'rounded')) else
                                      ('half_only' if gk.get((gold, 'half')) else None)))
            trap = sorted(set(b for (a, b) in t['rt_pairs'] if a == gold))
            row['trunc_trap'] = bool(trap)
            row['trunc_trap_letters'] = ''.join(trap)
            row['fell_in_trunc_trap'] = (t['pred'] in trap) if (trap and t['pred']) else None
            row['gold_seen_channels'] = sorted(ch for ch, c in t['recv_by_channel'].items() if gold in c)
            row['gold_first_seen_step'] = t['first_seen'].get(gold)
            row['pred_value_seen'] = (t['pred'] in recv) if t['pred'] else None
            row['pred_value_produced'] = (t['pred'] in produced) if t['pred'] else None
            rf = ref.get(it['id'], {})
            row['ref_options_present'] = ''.join(rf.get('letters', []))
            row['ref_gold_count'] = rf.get('gold_count')
            row['ref_unambiguous'] = (rf.get('letters') == [gold])
            row['traj_unambiguous'] = prod and not (produced - {gold})
            row['reexec_n'] = t['rex_n']
            row['reexec_fail'] = t['rex_fail']
            row['reexec_valid_last'] = t['reexec_valid']
            row['n_gold_lines_final'] = t['n_gold_lines']
            # read-range distance (result.out reads made after the LAST execution)
            if t['gold_lines']:
                le = t['last_exec_step']
                rngs = [(s, e) for (st, s, e, ex) in t['rr_reads'] if ex == le and s and e]
                if rngs:
                    dist = min(0 if s <= g <= e else min(abs(g - s), abs(g - e)) for g in t['gold_lines'] for (s, e) in rngs)
                    row['read_gap_lines_final'] = dist
            dl = t.get('default_letters')
            row['gold_in_default_print'] = (gold in dl) if dl is not None else None
            row['options_in_default_print'] = ''.join(sorted(dl)) if dl is not None else None
            if stage is None:
                if not prod:
                    stage = 'BUILD'
                elif not seen:
                    stage = 'NOT_SEEN'
                else:
                    stage = 'SEEN_NOT_USED'
            if not row['produced_flag_without_reexec'] and prod:
                notes.append('gold only found via offline re-execution')
            if t['rex_fail']:
                notes.append('%d re-exec failures/timeouts' % t['rex_fail'])
            if row['distractor_present']:
                notes.append('distractor value(s) %s also produced' % ''.join(sorted(produced - {gold})))
        else:
            row['seen_flag'] = None
        if not t['map_ok']:
            notes.append('question text did not match workspace log')
        if t['n_logs'] > 1:
            notes.append('%d chat logs; used %s' % (t['n_logs'], t['log_pick']))
        row['stage'] = stage
        row['notes'] = '; '.join(notes)
        out_rows.append(row)
    union45_ids = union45
    for row in out_rows:
        row['in_union45'] = row['item_id'] in union45_ids

    with open(os.path.join(SCRATCH, 'labels.jsonl'), 'w') as f:
        for row in out_rows:
            f.write(json.dumps(row) + '\n')
    with open(os.path.join(SCRATCH, 'items.json'), 'w') as f:
        json.dump({str(k): dict(id=v['id'], scenario=v['scenario'], kind=v['kind'], quantity=v['quantity'],
                                gold=v['gold'], gold_value=v['gold_value'], options=v['options'],
                                absmode=v.get('absmode'), intctx=v.get('intctx'), ref=ref.get(k))
                   for k, v in items.items()}, f, indent=1)
    print('[done] wrote %d label rows' % len(out_rows))
    # reexec validation stats
    vals = Counter(t.get('reexec_valid') for t in uniq.values() if items[t['item_id']]['kind'] == 'numeric')
    print('[reexec validation of last execution vs workspace result.out]', dict(vals))
    with open(os.path.join(SCRATCH, 'reexec_validation.json'), 'w') as f:
        json.dump({str(k): v for k, v in vals.items()}, f)
    summarize_main()


def summarize_main():
    write_summary(SCRATCH)


# ============================ summary.md =======================================

STAGES = ['CORRECT', 'NO_COMMIT', 'BUILD', 'NOT_SEEN', 'SEEN_NOT_USED', 'NONNUMERIC']
FAIL3 = ['BUILD', 'NOT_SEEN', 'SEEN_NOT_USED']
MODEL_ORDER = ['Claude Opus 4.6', 'Claude Sonnet 4.6', 'GPT-5.4', 'GPT-5.2', 'GPT-5.1', 'Gemini 3 Flash']
PROTO_ORDER = ['Raw100k', 'Hybrid', 'TOC', 'TOC-Oct', 'TOC+Report', 'TOC+Round', 'Grep', 'TOCfix45', 'TOC-capped']


def pct(a, b):
    return '%.1f' % (100.0 * a / b) if b else '-'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def fmt(x, d=2):
    return ('%.' + str(d) + 'f') % x if x == x else '-'


def sort_key(k):
    m, p = k[0], k[1]
    return (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, PROTO_ORDER.index(p) if p in PROTO_ORDER else 99) + tuple(k[2:])


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def stage_row(label, rs, numeric_only=False):
    n = len(rs)
    c = Counter(r['stage'] for r in rs)
    wrong_num = sum(c[s] for s in FAIL3)
    cells = [*label, n, pct(c['CORRECT'], n), c['NO_COMMIT']]
    if not numeric_only:
        cells.append(c['NONNUMERIC'])
    cells += [c['BUILD'], c['NOT_SEEN'], c['SEEN_NOT_USED'], wrong_num,
              pct(c['BUILD'], wrong_num), pct(c['NOT_SEEN'], wrong_num), pct(c['SEEN_NOT_USED'], wrong_num)]
    return cells


def write_summary(scratch):
    rows = [json.loads(l) for l in open(os.path.join(scratch, 'labels.jsonl'))]
    items = json.load(open(os.path.join(scratch, 'items.json')))
    try:
        val = json.load(open(os.path.join(scratch, 'reexec_validation.json')))
    except Exception:
        val = {}
    L = []
    w = L.append

    # groups -------------------------------------------------------------
    def grp(filter_fn, pool_toc=True):
        g = defaultdict(list)
        for r in rows:
            if not filter_fn(r):
                continue
            g[(r['model'], r['protocol'])].append(r)
        return g

    num = lambda r: r['numeric']
    num_unamb = lambda r: r['numeric'] and r.get('ref_unambiguous')
    every = lambda r: True

    kinds = Counter(v['kind'] for v in items.values())
    ml = Counter(r['source_dir'] for r in {(r['source_dir'], r['item_id']): r for r in rows}.values() if r['n_logs'] > 1)
    multi_log_note = (('workspaces with >1 log: ' + ', '.join('%s=%d' % kv for kv in ml.items())) if ml else
                      'no analysed workspace had >1 log; the 9 two-log workspaces of the GPT-5.1 TOC run2 are all '
                      'union45 items, which the splice replaces with the tocfix trajectory')
    ref_unamb = sum(1 for v in items.values() if v['kind'] == 'numeric' and v['ref'] and v['ref']['letters'] == [v['gold']])
    w('# Failure-stage decomposition of PHREEQC agent trajectories (158 verified items)\n')
    w('Generated by `analyze.py` (labels in `labels.jsonl`). All numbers restricted to the 158 `verified` items.\n')
    w('## 0. Method in one paragraph\n')
    w('For each (run, item) trajectory: **NO_COMMIT** = results.jsonl has no gradable prediction; **CORRECT**; for wrong '
      'answers on numeric items the keyed option value is searched (numerically, at the printed precision, see below) in '
      '(a) every tool result the agent actually received that carries PHREEQC output (inline `result_out_content`, '
      '`read_file` of result.out / selected-output files, `grep_file` match contexts) = **seen corpus**; and (b) the '
      '**produced corpus** = seen corpus + the final result.out / output files in the workspace + an offline '
      're-execution of *every* `execute_phreeqc` call (inputs reconstructed from the preceding `write_file` events, '
      'same PHREEQC 3.8.6 binary and phreeqc.dat). **BUILD** = gold value not in produced corpus; **NOT_SEEN** = '
      'produced but not seen; **SEEN_NOT_USED** = seen but answered wrong. Non-numeric items (categorical: '
      'More/Less, names; and "number of species" counts) are put in **NONNUMERIC** when wrong. '
      'Matching: a printed number matches an option if equal at the option\'s precision; a more precise printed '
      'number matches if it rounds to the option (in PHREEQC main output this requires >=1 extra digit in '
      'e-notation or >=2 extra digits in fixed notation, so that 3-decimal log-molalities do not match 2-decimal SI '
      'options; in selected-output files, or for fixed tokens with >=6 significant digits, any finer token may be '
      'rounded); a token exactly on a rounding midpoint (e.g. 7.9485e+03) matches both neighbours; '
      'for "dissolved / precipitated / delta" quantities the sign is ignored; integer specific-conductance options '
      'only match on lines containing "Conductance" or in an SC-named selected-output column. The input-deck echo '
      'in result.out is stripped before matching. TOC = the 3 reruns with the 45 cap-affected items replaced by the '
      'single uncapped `tocfix` trajectory (so the pooled TOC rows count each tocfix trajectory 3x, as in the '
      'paper\'s spliced numbers). Hybrid = every completed run under result/custom/hybrid158_verified (pooled). '
      'BUILD is further split by re-running the agent\'s own inputs with its PRINT / SELECTED_OUTPUT / USER_PUNCH '
      'blocks and TRANSPORT print options removed (BUILD/print-or-probe if the gold then appears, else '
      'BUILD/chemistry). Log selection: latest chat log containing a `run_end` event (%s).\n' % multi_log_note)
    w('Items: %d numeric, %d count-type ("number of X species"), %d categorical. Of the 137 numeric items, '
      '**%d are reference-unambiguous**: running the verification reference input for the scenario, only the gold '
      'option value occurs anywhere in the reference result.out (the others: a distractor value also occurs, '
      'typically at another reaction step / cell / temperature).\n' % (kinds['numeric'], kinds['count'], kinds['categorical'], ref_unamb))

    # 1. headline stage tables -------------------------------------------
    hdr_all = ['model', 'protocol', 'N', 'acc %', 'NO_COMMIT', 'wrong non-num', 'BUILD', 'NOT_SEEN', 'SEEN_NOT_USED',
               'wrong num', 'BUILD % of wrong-num', 'NOT_SEEN %', 'SEEN_NOT_USED %']
    hdr_num = ['model', 'protocol', 'N', 'acc %', 'NO_COMMIT', 'BUILD', 'NOT_SEEN', 'SEEN_NOT_USED',
               'wrong num', 'BUILD %', 'NOT_SEEN %', 'SEEN_NOT_USED %']
    for title, flt, numeric_only in [('all 158 verified items', every, False),
                                     ('numeric items (137)', num, True),
                                     ('numeric, reference-unambiguous items (%d)' % ref_unamb, num_unamb, True)]:
        g = grp(flt)
        w('## 1%s. Stage counts — %s\n' % ({'all 158 verified items': 'a', 'numeric items (137)': 'b'}.get(title, 'c'), title))
        w('TOC rows pool the 3 spliced reruns (3 x N items). Percent columns on the right are shares of the wrong numeric answers.\n')
        tr = []
        for k in sorted(g, key=sort_key):
            tr.append(stage_row(k, g[k], numeric_only))
        w(table(hdr_num if numeric_only else hdr_all, tr) + '\n')

    # 1d per rerun TOC
    w('## 1d. TOC per spliced rerun (all 158 items)\n')
    g = defaultdict(list)
    for r in rows:
        if r['protocol'] == 'TOC':
            g[(r['model'], r['protocol'], r['run'])].append(r)
    tr = [stage_row(k, g[k], False) for k in sorted(g, key=sort_key)]
    w(table(['model', 'protocol', 'run'] + hdr_all[2:], tr) + '\n')

    # 1e unique trajectories pooled
    w('## 1e. TOC pooled over unique trajectories (tocfix trajectory counted once)\n')
    g = defaultdict(dict)
    for r in rows:
        if r['protocol'] == 'TOC':
            g[(r['model'], r['protocol'])][(r['source_dir'], r['item_id'])] = r
    tr = [stage_row(k, list(g[k].values()), False) for k in sorted(g, key=sort_key)]
    w(table(hdr_all, tr) + '\n')

    # 2. share-of-all decomposition: where does the accuracy gap go? -------------
    w('## 2. Where does the Raw100k -> retrieval accuracy gap go? (stage shares of ALL trajectories, pp)\n')
    w('Each cell = share of trajectories in that stage (percent of N). Delta = retrieval minus Raw100k. A positive '
      'delta in NOT_SEEN would be the signature of an access failure.\n')
    for subset_name, flt in [('all 158 items', every), ('numeric, reference-unambiguous', num_unamb)]:
        w('**Subset: %s**\n' % subset_name)
        tr = []
        for model in ['GPT-5.1', 'GPT-5.2', 'Gemini 3 Flash']:
            base = [r for r in rows if r['model'] == model and r['protocol'] == 'Raw100k' and flt(r)]
            if not base:
                continue
            bs = Counter(r['stage'] for r in base)
            for proto in ['Raw100k', 'Hybrid', 'TOC', 'TOC-Oct', 'TOC+Report', 'Grep', 'TOC-capped']:
                rs = [r for r in rows if r['model'] == model and r['protocol'] == proto and flt(r)]
                if not rs:
                    continue
                c = Counter(r['stage'] for r in rs)
                cells = [model, proto, len(rs)]
                for s in STAGES:
                    v = 100.0 * c[s] / len(rs)
                    b = 100.0 * bs[s] / len(base)
                    cells.append('%.1f' % v if proto == 'Raw100k' else '%.1f (%+.1f)' % (v, v - b))
                tr.append(cells)
        w(table(['model', 'protocol', 'N'] + STAGES, tr) + '\n')
    # top-tier context
    tr = []
    for model in ['Claude Sonnet 4.6', 'GPT-5.4', 'Claude Opus 4.6']:
        for proto in ['TOC', 'Grep', 'TOCfix45']:
            rs = [r for r in rows if r['model'] == model and r['protocol'] == proto]
            if not rs:
                continue
            c = Counter(r['stage'] for r in rs)
            tr.append([model, proto, len(rs)] + ['%.1f' % (100.0 * c[s] / len(rs)) for s in STAGES])
    w('Top-tier models (no Raw100k run in this set), same shares:\n')
    w(table(['model', 'protocol', 'N'] + STAGES, tr) + '\n')

    # 2b. named-mechanism decomposition --------------------------------------
    def mech(r):
        st = r['stage']
        if st == 'CORRECT':
            return 'CORRECT'
        if st == 'NO_COMMIT':
            return 'NO_COMMIT/' + (r.get('no_commit_reason') or '?')
        if st == 'NONNUMERIC':
            return 'NONNUMERIC'
        if st == 'SEEN_NOT_USED':
            if r.get('fell_in_trunc_trap'):
                return 'SNU/truncated-high-precision'
            if r.get('gold_seen_kind') == 'half_only':
                return 'SNU/half-point-ambiguous'
            return 'SNU/other'
        if st == 'NOT_SEEN':
            saw_main = r['inline'] > 0 or r['n_read_resultout'] > 0 or r.get('n_grep_resultout', 0) > 0
            return 'NOT_SEEN/never-saw-result.out' if not saw_main else 'NOT_SEEN/read-elsewhere-in-result.out'
        if st == 'BUILD':
            g = r.get('gold_in_default_print')
            return 'BUILD/print-or-probe' if g else 'BUILD/chemistry'
        return st
    MECHS = ['CORRECT', 'BUILD/chemistry', 'BUILD/print-or-probe', 'NOT_SEEN/never-saw-result.out',
             'NOT_SEEN/read-elsewhere-in-result.out', 'SNU/truncated-high-precision', 'SNU/half-point-ambiguous', 'SNU/other',
             'NO_COMMIT/declined', 'NO_COMMIT/budget', 'NO_COMMIT/other', 'NONNUMERIC']
    w('## 2b. Named-mechanism decomposition (share of ALL trajectories, %; delta vs Raw100k in pp)\n')
    w('BUILD/print-or-probe = gold absent from everything the agent produced, but present when the agent\'s own '
      'final chemistry is re-run with its PRINT/SELECTED_OUTPUT/USER_PUNCH blocks and TRANSPORT print options '
      'removed (i.e. the chemistry was right but the requested printout omitted the asked quantity/step). '
      'SNU/truncated-high-precision = gold was on screen only via a high-precision (selected-output style) number '
      'that rounds to the gold option and the agent chose the option obtained by truncating it. '
      'SNU/half-point-ambiguous = gold on screen only as a 5-significant-digit selected-output value lying exactly '
      'on a rounding midpoint (e.g. 8.0415e-04 between options 8.041e-04 and 8.042e-04), so the printout itself '
      'cannot disambiguate. '
      'NOT_SEEN/never-saw-result.out = no inline payload, no read_file and no grep of result.out at all (answered '
      'from its own selected-output file).\n')
    for subset_name, flt in [('all 158 items', every), ('numeric, reference-unambiguous', num_unamb)]:
        w('**Subset: %s**\n' % subset_name)
        tr = []
        for model in ['GPT-5.1', 'GPT-5.2', 'GPT-5.4']:
            base = [r for r in rows if r['model'] == model and r['protocol'] == 'Raw100k' and flt(r)]
            bc = Counter(mech(r) for r in base)
            for proto in ['Raw100k', 'Hybrid', 'TOC', 'TOC-Oct', 'TOC+Report', 'Grep']:
                rs = [r for r in rows if r['model'] == model and r['protocol'] == proto and flt(r)]
                if not rs:
                    continue
                c = Counter(mech(r) for r in rs)
                cells = [model, proto, len(rs)]
                for mname in MECHS:
                    v = 100.0 * c[mname] / len(rs)
                    b = 100.0 * bc[mname] / len(base)
                    cells.append('%.1f' % v if proto == 'Raw100k' else '%.1f (%+.1f)' % (v, v - b))
                tr.append(cells)
        for model in ['Claude Sonnet 4.6', 'GPT-5.4', 'Claude Opus 4.6', 'Gemini 3 Flash']:
            for proto in ['TOC', 'Grep', 'TOCfix45', 'Raw100k', 'TOC-capped']:
                rs = [r for r in rows if r['model'] == model and r['protocol'] == proto and flt(r)]
                if not rs:
                    continue
                c = Counter(mech(r) for r in rs)
                tr.append([model, proto, len(rs)] + ['%.1f' % (100.0 * c[m_] / len(rs)) for m_ in MECHS])
        w(table(['model', 'protocol', 'N'] + MECHS, tr) + '\n')
    # raw counts version for GPT
    w('Counts behind the table (all 158 items):\n')
    tr = []
    for model in MODEL_ORDER:
        for proto in PROTO_ORDER:
            rs = [r for r in rows if r['model'] == model and r['protocol'] == proto]
            if not rs:
                continue
            c = Counter(mech(r) for r in rs)
            tr.append([model, proto, len(rs)] + [c[m_] for m_ in MECHS])
    w(table(['model', 'protocol', 'N'] + MECHS, tr) + '\n')

    # 2c. conditional pipeline rates + Shapley attribution of the gap -----------
    import random, itertools, math

    def rates(rs):
        n = len(rs)
        prod = [r for r in rs if r.get('produced_flag')]
        nprod = [r for r in rs if not r.get('produced_flag')]
        seen = [r for r in prod if r.get('seen_flag')]
        pns = [r for r in prod if not r.get('seen_flag')]
        cor = lambda L: sum(1 for r in L if r['stage'] == 'CORRECT')
        f = lambda a, b: a / b if b else 0.0
        return dict(n=n, p=f(len(prod), n), s=f(len(seen), len(prod)), c=f(cor(seen), len(seen)),
                    g=f(cor(pns), len(pns)), l=f(cor(nprod), len(nprod)), acc=f(cor(rs), n),
                    u=f(cor(pns) + cor(nprod), len(pns) + len(nprod)),
                    n_prod=len(prod), n_seen=len(seen), n_pns=len(pns), n_nprod=len(nprod))

    def acc_of(q):
        # exact identity: acc = P(seen)*P(correct|seen) + P(not seen)*P(correct|not seen)
        return q['p'] * q['s'] * q['c'] + (1 - q['p'] * q['s']) * q['u']

    FACT = {'build (p = P produced)': ['p'], 'access (s = P seen | produced)': ['s'],
            'closure (c = P correct | seen)': ['c'], 'unsupported answers (u = P correct | not seen)': ['u']}

    def shapley(base, alt):
        names = list(FACT)
        n = len(names)
        phi = {k: 0.0 for k in names}
        for k in names:
            others = [x for x in names if x != k]
            for r_ in range(len(others) + 1):
                for S in itertools.combinations(others, r_):
                    wgt = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                    q1 = dict(base); q2 = dict(base)
                    for x in S:
                        for v in FACT[x]:
                            q1[v] = alt[v]; q2[v] = alt[v]
                    for v in FACT[k]:
                        q2[v] = alt[v]
                    phi[k] += wgt * (acc_of(q2) - acc_of(q1))
        return phi

    w('## 2c. Pipeline view: conditional rates and a Shapley attribution of the Raw100k -> retrieval gap\n')
    w('Numeric items only. acc = p*s*c + (1-p*s)*u = p*s*c + p*(1-s)*g + (1-p)*l exactly, with p = P(gold produced by the agent\'s '
      'own simulation), s = P(gold reached the agent | produced), c = P(correct | seen), g = P(correct | produced '
      'but not seen), l = P(correct | not produced), u = P(correct | gold not seen) (g, l, u = answers not supported '
      'by an on-screen gold value: nearest-option picks among close distractors, lucky guesses). '
      'NO_COMMIT counts as not correct.\n')
    tr = []
    for model in MODEL_ORDER:
        for proto in PROTO_ORDER:
            rs = [r for r in rows if r['model'] == model and r['protocol'] == proto and r['numeric']]
            if not rs:
                continue
            q = rates(rs)
            tr.append([model, proto, q['n'], '%.1f' % (100 * q['acc']), '%.1f' % (100 * q['p']),
                       '%.1f (%d)' % (100 * q['s'], q['n_prod']), '%.1f (%d)' % (100 * q['c'], q['n_seen']),
                       '%.1f (%d)' % (100 * q['g'], q['n_pns']), '%.1f (%d)' % (100 * q['l'], q['n_nprod']),
                       '%.1f (%d)' % (100 * q['u'], q['n_pns'] + q['n_nprod'])])
    w(table(['model', 'protocol', 'N', 'acc %', 'p: produced %', 's: seen|produced % (n)', 'c: correct|seen % (n)',
             'g: correct|produced,unseen % (n)', 'l: correct|not produced % (n)', 'u: correct|not seen % (n)'], tr) + '\n')

    w('**Shapley attribution of the accuracy change (pp, numeric items) relative to the same model\'s Raw100k run**; '
      '95% intervals from 1000 item-cluster bootstrap resamples (items resampled jointly for both protocols).\n')
    shap_tbl = None
    for subset_name, sflt in [('all numeric items', num), ('numeric, reference-unambiguous items', num_unamb)]:
        random.seed(0)
        tr = []
        for model in ['GPT-5.1', 'GPT-5.2', 'GPT-5.4']:
            base_all = [r for r in rows if r['model'] == model and r['protocol'] == 'Raw100k' and sflt(r)]
            for proto in ['TOC', 'Grep', 'Hybrid', 'TOC-Oct', 'TOC+Report', 'TOC+Round']:
                alt_all = [r for r in rows if r['model'] == model and r['protocol'] == proto and sflt(r)]
                if not alt_all:
                    continue
                bq, aq = rates(base_all), rates(alt_all)
                phi = shapley(bq, aq)
                byb = defaultdict(list); bya = defaultdict(list)
                for r in base_all: byb[r['item_id']].append(r)
                for r in alt_all: bya[r['item_id']].append(r)
                ids = sorted(set(byb) & set(bya))
                boots = defaultdict(list)
                for _ in range(1000):
                    samp = [random.choice(ids) for _ in ids]
                    b_ = [r for i in samp for r in byb[i]]; a_ = [r for i in samp for r in bya[i]]
                    ph = shapley(rates(b_), rates(a_))
                    for k, v in ph.items():
                        boots[k].append(v)
                    boots['total'].append(rates(a_)['acc'] - rates(b_)['acc'])
                def ci(k):
                    xs = sorted(boots[k]); return '[%+.1f, %+.1f]' % (100 * xs[25], 100 * xs[974])
                cells = [model, proto, '%+.1f %s' % (100 * (aq['acc'] - bq['acc']), ci('total'))]
                for k in FACT:
                    cells.append('%+.1f %s' % (100 * phi[k], ci(k)))
                tr.append(cells)
        tbl = table(['model', 'protocol', 'total change'] + list(FACT), tr)
        if shap_tbl is None:
            shap_tbl = tbl
        w('*Subset: %s*\n' % subset_name)
        w(tbl + '\n')

    # 3. KEY comparison -----------------------------------------------------
    w('## 3. KEY comparison: trajectories wrong/no-commit under retrieval but CORRECT under Raw100k (same model, same item)\n')
    raw = {(r['model'], r['item_id']): r for r in rows if r['protocol'] == 'Raw100k'}
    for model in ['GPT-5.1', 'GPT-5.2', 'GPT-5.4']:
        for proto in ['TOC', 'Grep', 'Hybrid', 'TOC-Oct', 'TOC+Report', 'TOC+Round']:
            rs = [r for r in rows if r['model'] == model and r['protocol'] == proto
                  and r['stage'] != 'CORRECT' and raw.get((model, r['item_id']), {}).get('stage') == 'CORRECT']
            if not rs:
                continue
            uniq = {(r['source_dir'], r['item_id']): r for r in rs}
            c = Counter(r['stage'] for r in rs)
            cu = Counter(r['stage'] for r in uniq.values())
            nnum = [r for r in rs if r['numeric']]
            w('**%s, %s** — %d trajectory-rows (%d unique trajectories, %d distinct items) lost relative to Raw100k.\n' % (
                model, proto, len(rs), len(uniq), len(set(r['item_id'] for r in rs))))
            tr = []
            for s in ['NO_COMMIT', 'BUILD', 'NOT_SEEN', 'SEEN_NOT_USED', 'NONNUMERIC']:
                tr.append([s, c[s], pct(c[s], len(rs)), cu[s], pct(cu[s], len(uniq))])
            w(table(['stage', 'rows', '% rows', 'unique traj', '% unique'], tr) + '\n')
            mc = Counter(mech(r) for r in rs)
            w('Named mechanisms (rows): %s.\n' % ', '.join('%s=%d' % kv for kv in mc.most_common()))
            # NO_COMMIT reasons
            nc = Counter(r.get('no_commit_reason') for r in rs if r['stage'] == 'NO_COMMIT')
            if nc:
                w('NO_COMMIT reasons: %s.\n' % ', '.join('%s=%d' % kv for kv in nc.most_common()))
            # the same items under Raw
            rr = [raw[(model, r['item_id'])] for r in nnum]
            seen_raw = sum(1 for x in rr if x.get('seen_flag'))
            inline_raw = sum(1 for x in rr if 'inline' in (x.get('gold_seen_channels') or []))
            w('Same items under Raw100k (numeric rows, n=%d): gold value SEEN in %d (%s%%), of which via the inline '
              'execute_phreeqc payload in %d.\n' % (len(rr), seen_raw, pct(seen_raw, len(rr)), inline_raw))
            # within retrieval failures: did the retrieval run *produce* the gold?
            prod = sum(1 for r in nnum if r.get('produced_flag'))
            w('Within these %d numeric retrieval failures, the gold value was produced by the agent\'s own simulation in '
              '%d (%s%%) and reached the agent in %d (%s%%).\n' % (
                  len(nnum), prod, pct(prod, len(nnum)), sum(1 for r in nnum if r.get('seen_flag')),
                  pct(sum(1 for r in nnum if r.get('seen_flag')), len(nnum))))
    # converse
    w('**Converse (noise floor)**: trajectories correct under retrieval but wrong/no-commit under Raw100k.\n')
    tr = []
    for model in ['GPT-5.1', 'GPT-5.2', 'GPT-5.4']:
        for proto in ['TOC', 'Grep', 'Hybrid', 'TOC-Oct', 'TOC+Report', 'TOC+Round']:
            rs = [r for r in rows if r['model'] == model and r['protocol'] == proto]
            if not rs:
                continue
            lost = sum(1 for r in rs if r['stage'] != 'CORRECT' and raw.get((model, r['item_id']), {}).get('stage') == 'CORRECT')
            gained = sum(1 for r in rs if r['stage'] == 'CORRECT' and raw.get((model, r['item_id']), {}).get('stage') not in (None, 'CORRECT'))
            nrun = len(set(r['run'] for r in rs))
            tr.append([model, proto, nrun, lost, gained, '%.1f' % (100.0 * (gained - lost) / len(rs))])
    w(table(['model', 'protocol', '#runs pooled', 'lost vs Raw (rows)', 'gained vs Raw (rows)', 'net pp'], tr) + '\n')

    # 4. Sonnet TOC failures --------------------------------------------------
    w('## 4. Claude Sonnet 4.6 TOC failures (all non-correct trajectory-rows)\n')
    rs = [r for r in rows if r['model'] == 'Claude Sonnet 4.6' and r['protocol'] == 'TOC' and r['stage'] != 'CORRECT']
    c = Counter(r['stage'] for r in rs)
    w('Counts: %s (of %d rows).\n' % (', '.join('%s=%d' % kv for kv in c.most_common()), sum(1 for r in rows if r['model'] == 'Claude Sonnet 4.6' and r['protocol'] == 'TOC')))
    tr = []
    for r in sorted(rs, key=lambda r: (r['item_id'], r['run'])):
        it = items[str(r['item_id'])]
        tr.append([r['item_id'], r['run'], r['source'], it['quantity'][:45], r['gold'], r['pred'], r['stage'],
                   r.get('options_present', ''), r.get('options_seen', ''), r['n_read_resultout']])
    w(table(['item', 'run', 'src', 'quantity', 'gold', 'pred', 'stage', 'opts produced', 'opts seen', '#read result.out'], tr) + '\n')
    gr = [r for r in rows if r['model'] == 'Claude Sonnet 4.6' and r['protocol'] == 'Grep' and r['stage'] != 'CORRECT']
    w('Sonnet Grep: %s.\n' % ', '.join('%s=%d' % kv for kv in Counter(r['stage'] for r in gr).most_common()))

    # 5. process metrics -----------------------------------------------------
    w('## 5. Process metrics (means per trajectory)\n')
    keys = ['steps', 'n_tool', 'n_exec', 'n_read', 'n_read_output', 'n_read_resultout', 'n_read_other_output', 'n_grep']
    for title, flt in [('all trajectories', lambda r: True), ('failures only (wrong or NO_COMMIT)', lambda r: r['stage'] != 'CORRECT'),
                       ('correct only', lambda r: r['stage'] == 'CORRECT')]:
        w('**%s**\n' % title)
        g = defaultdict(list)
        for r in rows:
            if flt(r):
                g[(r['model'], r['protocol'])].append(r)
        tr = []
        for k in sorted(g, key=sort_key):
            rs = g[k]
            tr.append([k[0], rs[0]['tier'], k[1], len(rs)] + [fmt(mean([r[x] for r in rs])) for x in keys])
        w(table(['model', 'tier', 'protocol', 'N'] + keys, tr) + '\n')
    # tier aggregate
    w('**Tier aggregate (all trajectories; TOC pooled)**\n')
    g = defaultdict(list)
    for r in rows:
        g[(r['tier'], r['protocol'])].append(r)
    tr = []
    for k in sorted(g):
        rs = g[k]
        tr.append([k[0], k[1], len(rs)] + [fmt(mean([r[x] for r in rs])) for x in keys])
    w(table(['tier', 'protocol', 'N'] + keys, tr) + '\n')

    # 6. specific fractions -------------------------------------------------
    w('## 6. Failure-specific fractions\n')
    tr = []
    for model in MODEL_ORDER:
        for proto in ['TOC', 'TOCfix45', 'TOC-capped', 'Grep', 'Hybrid', 'Raw100k']:
            rs = [r for r in rows if r['model'] == model and r['protocol'] == proto]
            if not rs:
                continue
            fails = [r for r in rs if r['stage'] != 'CORRECT']
            corr = [r for r in rs if r['stage'] == 'CORRECT']
            if proto.startswith('TOC') or proto == 'Hybrid':
                f_toc = [r for r in fails if r['toc_nonempty'] > 0]
                z_ro = sum(1 for r in f_toc if r['n_read_resultout'] == 0)
                z_any = sum(1 for r in f_toc if r['n_read_output'] == 0)
                c_toc = [r for r in corr if r['toc_nonempty'] > 0]
                cz = sum(1 for r in c_toc if r['n_read_resultout'] == 0)
                zcell = '%d/%d (%s%%)' % (z_ro, len(f_toc), pct(z_ro, len(f_toc)))
                zany = '%d/%d (%s%%)' % (z_any, len(f_toc), pct(z_any, len(f_toc)))
                zc = '%d/%d (%s%%)' % (cz, len(c_toc), pct(cz, len(c_toc)))
            elif proto == 'Grep':
                z_ro = sum(1 for r in fails if r['n_read_resultout'] == 0 and r['n_grep'] == 0)
                z_any = sum(1 for r in fails if r['n_read_output'] == 0 and r['n_grep'] == 0)
                cz = sum(1 for r in corr if r['n_read_resultout'] == 0 and r['n_grep'] == 0)
                zcell = '%d/%d (%s%%) [no grep+no read]' % (z_ro, len(fails), pct(z_ro, len(fails)))
                zany = '%d/%d (%s%%)' % (z_any, len(fails), pct(z_any, len(fails)))
                zc = '%d/%d (%s%%)' % (cz, len(corr), pct(cz, len(corr)))
            else:
                zcell = zany = zc = 'n/a'
            wn = [r for r in rs if r['stage'] in FAIL3]
            pd = sum(1 for r in wn if r.get('pred_value_seen'))
            pd_snu = sum(1 for r in wn if r.get('pred_value_seen') and r['stage'] == 'SEEN_NOT_USED')
            pd_prod = sum(1 for r in wn if r.get('pred_value_produced'))
            tr.append([model, proto, len(fails), zcell, zany, zc,
                       '%d/%d (%s%%)' % (pd, len(wn), pct(pd, len(wn))), pd_snu,
                       '%d/%d (%s%%)' % (pd_prod, len(wn), pct(pd_prod, len(wn)))])
    w('Columns: (i) failures (wrong or NO_COMMIT) that received a non-empty TOC/large-output index but issued zero '
      'read_file calls on result.out; (ii) same but zero reads of ANY output file (result.out or selected-output); '
      '(iii) the same zero-result.out-read rate among correct trajectories for comparison; (iv) wrong numeric answers '
      'whose chosen option value appears in a tool result the agent saw (= picked a distractor that was on screen), '
      'and how many of those are SEEN_NOT_USED (gold also on screen); (v) chosen option value produced anywhere.\n')
    w(table(['model', 'protocol', '#fail', '(i) TOC rcvd, 0 result.out reads', '(ii) 0 reads of any output',
             '(iii) correct: 0 result.out reads', '(iv) wrong-num: chosen value seen', 'of which SNU', '(v) chosen value produced'], tr) + '\n')

    # seen channels for correct items (base rate)
    w('## 7. SEEN base rate on CORRECT numeric trajectories and channel of first contact\n')
    tr = []
    g = defaultdict(list)
    for r in rows:
        if r['numeric'] and r['stage'] == 'CORRECT':
            g[(r['model'], r['protocol'])].append(r)
    for k in sorted(g, key=sort_key):
        rs = g[k]
        s = sum(1 for r in rs if r.get('seen_flag'))
        ch = Counter()
        for r in rs:
            for c_ in r.get('gold_seen_channels') or []:
                ch[c_] += 1
        tr.append([k[0], k[1], len(rs), '%s' % pct(s, len(rs)), ', '.join('%s=%d' % kv for kv in ch.most_common())])
    w('If the gold value was printed and received, SEEN should be ~100% on correct trajectories; the shortfall is '
      'the false-negative rate of the matcher plus genuine lucky guesses.\n')
    w(table(['model', 'protocol', 'N correct numeric', 'gold seen %', 'channels where gold seen (count of trajectories)'], tr) + '\n')

    # 8. reliability --------------------------------------------------------
    w('## 8. Reliability of the labels\n')
    numrows = [r for r in rows if r['numeric']]
    wr = [r for r in numrows if r['stage'] in FAIL3]
    w('- Numeric trajectory-rows: %d; wrong numeric: %d.' % (len(numrows), len(wr)))
    dp = sum(1 for r in numrows if r.get('distractor_present'))
    w('- Distractor value present somewhere in the agent\'s produced output: %d/%d numeric rows (%s%%); among wrong numeric %d/%d (%s%%).' % (
        dp, len(numrows), pct(dp, len(numrows)), sum(1 for r in wr if r.get('distractor_present')), len(wr),
        pct(sum(1 for r in wr if r.get('distractor_present')), len(wr))))
    nop = Counter(r.get('n_options_present') for r in numrows)
    w('- Distribution of n_options_present (0..4) over numeric rows: %s.' % ', '.join('%s:%d' % (k, nop[k]) for k in sorted(nop)))
    nopw = Counter(r.get('n_options_present') for r in wr)
    w('- ... over wrong numeric rows: %s.' % ', '.join('%s:%d' % (k, nopw[k]) for k in sorted(nopw)))
    flip = sum(1 for r in wr if r.get('produced_flag') and not r.get('produced_flag_without_reexec'))
    w('- Wrong numeric rows moved BUILD -> NOT_SEEN by the offline re-execution (gold only in an execution whose '
      'output was never shown and later overwritten): %d.' % flip)
    w('- Re-execution validation (last execution re-run vs the workspace result.out, path lines stripped): %s.' % val)
    rf = sum(r.get('reexec_fail', 0) for r in numrows)
    w('- Re-executions that timed out / could not run: %d (counted over trajectory rows).' % rf)
    tot = len(rows)
    nc = sum(1 for r in rows if r['stage'] == 'NO_COMMIT')
    nn = sum(1 for r in rows if r['stage'] == 'NONNUMERIC')
    fails = sum(1 for r in rows if r['stage'] != 'CORRECT')
    w('- Unclassifiable failures: NO_COMMIT %d + NONNUMERIC %d = %d of %d failures (%s%%) across all rows.' % (
        nc, nn, nc + nn, fails, pct(nc + nn, fails)))
    ok = Counter(r['map_ok'] for r in rows)
    w('- Workspace<->item mapping verified via question text in the first user message: %s.' % dict(ok))
    w('')
    # BUILD near-miss: options present in BUILD rows
    b = [r for r in wr if r['stage'] == 'BUILD']
    bo = Counter(r.get('n_options_present') for r in b)
    w('- BUILD rows by number of option values the agent\'s simulation produced: %s (0 = the agent\'s simulation '
      'produced none of the four values, i.e. a different system; >=1 = it produced a distractor value).\n' % ', '.join('%s:%d' % (k, bo[k]) for k in sorted(bo)))

    # 8b. mechanism: selected-output reliance and the truncation trap ------------
    w('## 8b. Mechanism probe: SELECTED_OUTPUT reliance and the rounding/truncation trap\n')
    w('PHREEQC\'s main tables print values already rounded to the precision used by the options (e.g. SI to 2 '
      'decimals, pH to 3). SELECTED_OUTPUT/USER_PUNCH files print more digits (e.g. si = -0.4469), so an agent that '
      'reads those must round itself. "Trap" = the agent received a high-precision token that ROUNDS to the gold '
      'option but TRUNCATES to a distractor option; "fell" = the agent chose exactly that distractor.\n')
    tr = []
    for model in MODEL_ORDER:
        for proto in ['Raw100k', 'Hybrid', 'TOC', 'TOC-Oct', 'TOC+Report', 'TOC+Round', 'Grep', 'TOCfix45', 'TOC-capped']:
            rs = [r for r in rows if r['model'] == model and r['protocol'] == proto]
            if not rs:
                continue
            n = len(rs)
            seld = sum(1 for r in rs if r.get('sel_defined'))
            rdsel = sum(1 for r in rs if r['n_read_other_output'] > 0)
            z_ro = sum(1 for r in rs if r['n_read_resultout'] == 0 and r['inline'] == 0 and r.get('n_grep_resultout', 0) == 0)
            nr = [r for r in rs if r['numeric'] and r.get('seen_flag')]
            ro = sum(1 for r in nr if r.get('gold_seen_kind') in ('rounded_only', 'half_only'))
            trap = [r for r in rs if r['numeric'] and r.get('trunc_trap') and r['stage'] != 'NO_COMMIT']
            fell = sum(1 for r in trap if r.get('fell_in_trunc_trap'))
            trap_ok = sum(1 for r in trap if r['stage'] == 'CORRECT')
            snu = [r for r in rs if r['stage'] == 'SEEN_NOT_USED']
            snu_tr = sum(1 for r in snu if r.get('fell_in_trunc_trap'))
            tr.append([model, proto, n, pct(seld, n), pct(rdsel, n), pct(z_ro, n),
                       '%d/%d (%s%%)' % (ro, len(nr), pct(ro, len(nr))),
                       '%d: fell %d (%s%%), correct %d' % (len(trap), fell, pct(fell, len(trap)), trap_ok),
                       '%d/%d' % (snu_tr, len(snu))])
    w(table(['model', 'protocol', 'N', '% defined SELECTED_OUTPUT/USER_PUNCH', '% read a selected-output file',
             '% never saw result.out content (no inline, no read, no grep)', 'gold seen only as high-precision token (numeric, seen)',
             'trunc-trap trajectories', 'SEEN_NOT_USED explained by trap'], tr) + '\n')

    # 8c. NO_COMMIT anatomy ----------------------------------------------------
    w('## 8c. NO_COMMIT anatomy (numeric items): was the gold produced / seen before the agent gave up?\n')
    tr = []
    for model in MODEL_ORDER:
        for proto in ['Raw100k', 'Hybrid', 'TOC', 'TOC-Oct', 'TOC+Report', 'TOC+Round', 'Grep', 'TOCfix45', 'TOC-capped']:
            rs = [r for r in rows if r['model'] == model and r['protocol'] == proto and r['stage'] == 'NO_COMMIT']
            if not rs:
                continue
            reasons = Counter(r.get('no_commit_reason') for r in rs)
            nr = [r for r in rs if r['numeric']]
            tr.append([model, proto, len(rs), ', '.join('%s=%d' % kv for kv in reasons.most_common()), len(nr),
                       sum(1 for r in nr if not r.get('produced_flag')),
                       sum(1 for r in nr if r.get('produced_flag') and not r.get('seen_flag')),
                       sum(1 for r in nr if r.get('seen_flag')), fmt(mean([r['steps'] for r in rs]), 1)])
    w(table(['model', 'protocol', 'NO_COMMIT', 'reasons', 'numeric', 'gold not produced', 'produced not seen', 'seen', 'mean steps'], tr) + '\n')

    # 9. tocfix-45 only -----------------------------------------------------
    w('## 9. The 45 cap-affected (tocfix) items alone — verified subset\n')
    g = defaultdict(dict)
    for r in rows:
        if r['protocol'] in ('TOC', 'TOCfix45') and r['in_union45']:
            g[(r['model'], 'TOCfix45')][(r['source_dir'], r['item_id'])] = r
        elif r['in_union45'] and r['protocol'] in ('Raw100k', 'Grep', 'Hybrid', 'TOC-capped'):
            g[(r['model'], r['protocol'] + ' (same items)')][(r['source_dir'], r['item_id'], r['run'])] = r
    tr = [stage_row(k, list(g[k].values()), False) for k in sorted(g, key=sort_key)]
    w(table(hdr_all, tr) + '\n')

    # 10. read-gap for NOT_SEEN --------------------------------------------
    w('## 10. NOT_SEEN detail: how far were the agent\'s result.out reads from the gold line?\n')
    tr = []
    for r in sorted([r for r in rows if r['stage'] == 'NOT_SEEN'], key=lambda r: (MODEL_ORDER.index(r['model']), r['protocol'], r['item_id'], r['run'])):
        tr.append([r['model'], r['protocol'], r['run'], r['item_id'], items[str(r['item_id'])]['quantity'][:40],
                   r['gold'], r['pred'], r.get('gold_in_final_resultout'), r.get('read_gap_lines_final', ''),
                   r['n_read_resultout'], r['n_read_other_output'], r['n_grep'], r.get('options_seen', '')])
    w(table(['model', 'protocol', 'run', 'item', 'quantity', 'gold', 'pred', 'gold in final result.out',
             'min line gap of reads after last exec', '#read result.out', '#read other out', '#grep', 'opts seen'], tr) + '\n')

    # headline block (auto-computed) inserted right after the title
    head = ['## Headline (auto-computed; details in sections 2c, 2b, 8b)\n',
            'Shapley attribution of the GPT-5.1 / GPT-5.2 accuracy change vs their Raw100k run (numeric items, pp, '
            '95% item-bootstrap intervals). build = the agent\'s own simulation printed the keyed value somewhere; '
            'access = that value reached the agent in a tool result; closure = agent answered correctly given it saw '
            'the value; unsupported = correctness when the value was never on screen (nearest-option picks / guesses, '
            'NO_COMMIT counts as wrong).\n', shap_tbl + '\n']
    L[2:2] = head
    with open(os.path.join(scratch, 'summary.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote summary.md')



if __name__ == '__main__':
    main()
