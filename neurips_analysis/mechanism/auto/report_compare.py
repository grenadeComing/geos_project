"""Compare the October report-reminder runs (TOC+Report) with the same-day TOC control (TOC-Oct),
the May TOC runs (unique trajectories), the July inline run, and the October Hybrid runs.

Usage: python3 report_compare.py <labels.jsonl> <repo_root>
"""
import csv, json, random, sys
from collections import defaultdict, Counter

lab_path, repo = sys.argv[1], sys.argv[2]
ver = set(int(json.loads(l)['id']) for l in open(repo + '/datasets/neurips_dataset_158_verified.jsonl'))
rows = [json.loads(l) for l in open(lab_path)]

# unique trajectories per (model, protocol): the May TOC rows repeat each tocfix trajectory 3x
U = defaultdict(dict)
for d in rows:
    U[(d['model'], d['protocol'])][(d['source_dir'], d['item_id'])] = d
head = defaultdict(dict)
for r in csv.DictReader(open(repo + '/neurips_analysis/baseline_snapshot/spliced_per_item_correctness.csv')):
    if int(r['item_id']) in ver:
        head[r['model']][int(r['item_id'])] = r

MODELS = ['GPT-5.1', 'GPT-5.2']
PROTOS = ['Raw100k', 'TOC', 'TOC-Oct', 'TOC+Report', 'Hybrid']


def pct(a, b):
    return '%5.1f' % (100.0 * a / b) if b else '    -'


def skipped(d):
    return d['n_read_resultout'] == 0 and d['inline'] == 0 and d.get('n_grep_resultout', 0) == 0


print('== Per-protocol process measures (unique trajectories, 158 verified items) ==')
print('model    protocol     runs  N    acc   skip  readSel | numeric: p     s|p   c|s  | seen: exact%  acc|exact  acc|unrounded | trap fell')
for m in MODELS:
    for p in PROTOS:
        rs = list(U[(m, p)].values())
        if not rs:
            continue
        runs = len(set(d['source_dir'] for d in rs))
        n = len(rs)
        acc = sum(1 for d in rs if d['correct'])
        sk = sum(1 for d in rs if skipped(d))
        rsel = sum(1 for d in rs if d['n_read_other_output'] > 0)
        num = [d for d in rs if d['numeric']]
        prod = [d for d in num if d['produced_flag']]
        seen = [d for d in prod if d['seen_flag']]
        cor = [d for d in seen if d['correct']]
        ex = [d for d in seen if d['gold_seen_kind'] == 'exact']
        un = [d for d in seen if d['gold_seen_kind'] in ('rounded_only', 'half_only')]
        trap = [d for d in num if d.get('trunc_trap') and d['stage'] != 'NO_COMMIT']
        fell = [d for d in trap if d.get('fell_in_trunc_trap')]
        print('%-8s %-12s %3d %5d %s %s %s   | %s %s %s | %s %s %s  (n=%d/%d) | %d/%d' % (
            m, p, runs, n, pct(acc, n), pct(sk, n), pct(rsel, n),
            pct(len(prod), len(num)), pct(len(seen), len(prod)), pct(len(cor), len(seen)),
            pct(len(ex), len(seen)), pct(sum(1 for d in ex if d['correct']), len(ex)),
            pct(sum(1 for d in un if d['correct']), len(un)), len(ex), len(un), len(fell), len(trap)))
    print()

print('== Where correct numeric answers saw the keyed value (channel shares among correct, seen) ==')
for m in MODELS:
    for p in PROTOS:
        rs = [d for d in U[(m, p)].values() if d['numeric'] and d['correct'] and d['seen_flag']]
        if not rs:
            continue
        c = Counter()
        for d in rs:
            ch = set(d.get('gold_seen_channels') or [])
            c['report(read/inline/grep)' if ch & {'read_main', 'inline', 'grep'} else 'selected-output only'] += 1
        print('%-8s %-12s n=%3d  %s' % (m, p, len(rs), ', '.join('%s %s%%' % (k, pct(v, len(rs)).strip()) for k, v in sorted(c.items()))))
print()

random.seed(7)


def boot_ci(vals, B=4000):
    n = len(vals)
    out = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n * 100 for _ in range(B))
    return out[int(0.025 * B)], out[int(0.975 * B) - 1]


def per_item(m, p):
    g = defaultdict(list)
    for d in U[(m, p)].values():
        g[d['item_id']].append(int(bool(d['correct'])))
    if p == 'Raw100k':
        for i in ver:
            if i in head[m]:
                g[i].append(int(head[m][i]['raw100k']))
    if p == 'TOC':
        for i in ver:
            if i in head[m]:
                g[i].append(int(head[m][i]['toc_new']))
    return g


print('== Paired item-level differences (per-item mean accuracy; item bootstrap 95% CI) ==')
print('   Raw100k = 2 inline runs (May headline + July logged); TOC = May headline + 3 reruns (unique trajectories)')
for m in MODELS:
    A = {p: per_item(m, p) for p in PROTOS}
    for a, b in [('TOC+Report', 'TOC-Oct'), ('TOC-Oct', 'TOC'), ('TOC-Oct', 'Raw100k'), ('TOC+Report', 'Raw100k'),
                 ('TOC', 'Raw100k'), ('Hybrid', 'Raw100k'), ('TOC+Report', 'Hybrid')]:
        ids = sorted(i for i in ver if A[a].get(i) and A[b].get(i))
        if not ids:
            continue
        dv = [sum(A[a][i]) / len(A[a][i]) - sum(A[b][i]) / len(A[b][i]) for i in ids]
        lo, hi = boot_ci(dv)
        print('%-8s %-11s - %-8s  %+5.1f pp [%+5.1f, %+5.1f]  (n_items=%d)' % (m, a, b, 100 * sum(dv) / len(dv), lo, hi, len(ids)))
    print()
