import json, random, csv, itertools, sys
from collections import defaultdict, Counter
lab, repo = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(lab)]
ver = set(int(json.loads(l)['id']) for l in open(repo + '/datasets/neurips_dataset_158_verified.jsonl'))
U = defaultdict(dict); R = defaultdict(list)
for d in rows:
    U[(d['model'], d['protocol'])][(d['source_dir'], d['item_id'])] = d
    R[(d['model'], d['protocol'])].append(d)
scen = {d['item_id']: d['scenario'] for d in rows}

# (d) both forms in view
print('== TOC trajectories that saw the keyed value both as printed and with more digits')
for m in ['GPT-5.4', 'GPT-5.2', 'GPT-5.1']:
    for p in ['TOC', 'TOC-Oct']:
        us = [d for d in U[(m, p)].values() if d['numeric'] and d.get('seen_flag')]
        both = [d for d in us if d['gold_seen_kind'] == 'exact' and (set(d.get('gold_seen_channels') or []) & {'read_sel', 'grep_sel'})]
        ex_only = [d for d in us if d['gold_seen_kind'] == 'exact' and not (set(d.get('gold_seen_channels') or []) & {'read_sel', 'grep_sel'})]
        un = [d for d in us if d['gold_seen_kind'] in ('rounded_only', 'half_only')]
        f = lambda g: '%d/%d=%.1f%%' % (sum(1 for d in g if d['correct']), len(g), 100*sum(1 for d in g if d['correct'])/len(g)) if g else '-'
        print(m, p, 'both:', f(both), ' printed only:', f(ex_only), ' digits only:', f(un))

# (a) semantic: items where the keyed value occurs exactly once in the reference report
cnt = Counter()
once = set()
for d in rows:
    if d['numeric']:
        cnt[d['item_id']] = d.get('ref_gold_count')
for i, c in cnt.items():
    if c == 1: once.add(i)
print('\n== numeric items:', len(cnt), ' keyed value occurs exactly once in the reference report:', len(once), Counter(cnt.values()).most_common(6))

def rates(rs):
    n = len(rs); P = [d for d in rs if d['produced_flag']]; S = [d for d in P if d['seen_flag']]
    p = len(P)/n; s = len(S)/len(P) if P else 0; c = sum(1 for d in S if d['correct'])/len(S) if S else 0
    NS = [d for d in rs if not d['seen_flag']]; u = sum(1 for d in NS if d['correct'])/len(NS) if NS else 0
    return dict(p=p, s=s, c=c, u=u)
def acc(q): return q['p']*q['s']*q['c'] + (1 - q['p']*q['s'])*q['u']
def shapley(b, a):
    keys = ['p', 's', 'c', 'u']; phi = dict.fromkeys(keys, 0.0); perms = list(itertools.permutations(keys))
    for order in perms:
        cur = dict(b)
        for k in order:
            before = acc(cur); cur[k] = a[k]; phi[k] += (acc(cur) - before)/len(perms)
    return phi
def subset_shapley(m, p, items, B=1000):
    base = [d for d in R[(m, 'Raw100k')] if d['numeric'] and d['item_id'] in items]
    alt = [d for d in R[(m, p)] if d['numeric'] and d['item_id'] in items]
    pt = shapley(rates(base), rates(alt))
    bi = defaultdict(list); ai = defaultdict(list)
    for d in base: bi[d['item_id']].append(d)
    for d in alt: ai[d['item_id']].append(d)
    ids = sorted(set(bi) & set(ai)); random.seed(5); res = defaultdict(list)
    for _ in range(B):
        smp = [ids[random.randrange(len(ids))] for _ in ids]
        b = [d for i in smp for d in bi[i]]; a = [d for i in smp for d in ai[i]]
        try:
            ph = shapley(rates(b), rates(a))
        except ZeroDivisionError:
            continue
        for k, v in ph.items(): res[k].append(v)
    ci = {k: (sorted(v)[int(.025*len(v))]*100, sorted(v)[int(.975*len(v))-1]*100) for k, v in res.items()}
    return pt, ci, len(ids)
print('Shapley vs inline on items whose keyed value occurs once (pp, 95% item bootstrap):')
for m, p in [('GPT-5.2', 'TOC'), ('GPT-5.1', 'TOC'), ('GPT-5.4', 'TOC'), ('GPT-5.2', 'TOC-Oct'), ('GPT-5.1', 'TOC-Oct')]:
    pt, ci, n = subset_shapley(m, p, once)
    print('%s %s n=%d  build %+.1f [%+.1f,%+.1f]  exposure %+.1f [%+.1f,%+.1f]  closure %+.1f [%+.1f,%+.1f]' % (
        m, p, n, pt['p']*100, *ci['p'], pt['s']*100, *ci['s'], pt['c']*100, *ci['c']))

# (b) setup-clustered bootstrap for the primary contrast
head = defaultdict(dict)
for r in csv.DictReader(open(repo + '/neurips_analysis/baseline_snapshot/spliced_per_item_correctness.csv')):
    if int(r['item_id']) in ver: head[r['model']][int(r['item_id'])] = r
def per_item(m, protos, cols):
    g = defaultdict(list)
    for p in protos:
        for d in R[(m, p)]: g[d['item_id']].append(int(bool(d['correct'])))
    for c in cols:
        for i in ver: g[i].append(int(head[m][i][c]))
    return g
spec = {'Claude Opus 4.6': ([], ['toc_new'], [], ['raw100k']), 'Claude Sonnet 4.6': (['TOC'], ['toc_new'], [], ['raw100k']),
        'GPT-5.4': (['TOC'], ['toc_new'], ['Raw100k'], ['raw100k']), 'GPT-5.2': (['TOC', 'TOC-Oct'], ['toc_new'], ['Raw100k'], ['raw100k']),
        'GPT-5.1': (['TOC', 'TOC-Oct'], ['toc_new'], ['Raw100k'], ['raw100k'])}
print('\n== primary contrast with setup-clustered bootstrap (19 setups resampled whole)')
random.seed(11)
for m, (tp, tc, rp, rc) in spec.items():
    a, b = per_item(m, tp, tc), per_item(m, rp, rc)
    ids = sorted(i for i in ver if a.get(i) and b.get(i))
    dv = {i: sum(a[i])/len(a[i]) - sum(b[i])/len(b[i]) for i in ids}
    by = defaultdict(list)
    for i in ids: by[scen.get(i, '?')].append(i)
    keys = sorted(by); out = []
    for _ in range(4000):
        smp = [keys[random.randrange(len(keys))] for _ in keys]
        its = [i for k in smp for i in by[k]]
        out.append(100*sum(dv[i] for i in its)/len(its))
    out.sort()
    loo = [100*sum(dv[i] for i in ids if scen.get(i) != k)/sum(1 for i in ids if scen.get(i) != k) for k in keys]
    print('%-18s %+.1f  setup CI [%+.1f, %+.1f]  setups=%d  leave-one-setup-out range [%+.1f, %+.1f]' % (
        m, 100*sum(dv.values())/len(dv), out[100], out[3899], len(keys), min(loo), max(loo)))

# (c) GPT-5.4 on one basis
r4 = R[('GPT-5.4', 'TOC')]; raw4 = U[('GPT-5.4', 'Raw100k')].values()
print('\n== GPT-5.4 same basis (three TOC reruns vs logged inline run): all items %.1f vs %.1f' % (
    100*sum(1 for d in r4 if d['correct'])/len(r4), 100*sum(1 for d in raw4 if d['correct'])/len(list(raw4))))
