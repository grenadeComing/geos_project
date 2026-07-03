"""Spliced hybrid-protocol prediction (NO new runs; rebuttal-prep estimate).

Hybrid protocol definition: execute_phreeqc returns raw output inline when
result.out <= 100k chars, and a TOC otherwise. Predicted per-item outcome is
spliced from two completed runs: Raw100k correctness on small-output items,
spliced-TOC correctness on long-output items.

Three partition variants (which items count as "long-output"):
  A own    - the model's own TOC-run output sizes (per_item_output_size_per_model.csv)
  B union45 - 5-model union of cap-affected items (cap_affected_union45_ids.txt)
  C canon   - canonical Sonnet run1 reference sizes (canonical_output_sizes_sonnet_run1.csv)

!!! DOUBLE-CHECK BEFORE USING IN REBUTTAL/PAPER !!!
  1. Between-run splice, not within-trajectory switching: a real hybrid agent can
     see both inline and TOC returns inside one trajectory; behavior may differ.
  2. Partition sensitivity is real: for GPT-5.1 variant A (63.0%) lands below pure
     Raw while B/C (65.0/66.0%) land above -- the sign flips with the partition.
  3. Opus 4.6 is MISSING from per_item_output_size_per_model.csv, so its variant A
     degenerates to pure-Raw; ignore Opus/hyA.
  4. canonical_output_sizes_sonnet_run1.csv has 38 cap-affected items, but the
     paper's canonical reference reports 37 -- reconcile which run/threshold
     before quoting variant C.
  5. Inherits single-run noise from both source runs; GPT-5.1's hybrid-vs-Raw
     margin is 3-5 items on a 45-item slice (not significant on its own).

Run:  python hybrid_splice_prediction.py   (from repo root or this directory)
Outputs hybrid_splice_prediction.csv next to this script.
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda name: os.path.join(HERE, name)

corr = defaultdict(dict)
for r in csv.DictReader(open(p('spliced_per_item_correctness.csv'))):
    if r['toc_new'] != '' and r['raw100k'] != '':
        corr[r['model']][int(r['item_id'])] = (int(r['toc_new']), int(r['raw100k']))

union45 = {int(x) for x in open(p('cap_affected_union45_ids.txt')).read().split()}
canon = {int(r['item_id'])
         for r in csv.DictReader(open(p('canonical_output_sizes_sonnet_run1.csv')))
         if r['cap_affected'] == '1'}
permodel = defaultdict(set)
for r in csv.DictReader(open(p('per_item_output_size_per_model.csv'))):
    if r['cap_affected'] == '1':
        permodel[r['model']].add(int(r['item_id']))

rows = []
for model, items in sorted(corr.items()):
    assert len(items) == 200, (model, len(items))
    toc = sum(v[0] for v in items.values())
    raw = sum(v[1] for v in items.values())

    def hybrid(longset):
        return sum((v[0] if i in longset else v[1]) for i, v in items.items())

    own = permodel.get(model)
    rows.append({
        'model': model,
        'toc_pct': toc / 2, 'raw100k_pct': raw / 2,
        'hybrid_own_pct': hybrid(own) / 2 if own else '',  # blank: no size data (Opus)
        'hybrid_union45_pct': hybrid(union45) / 2,
        'hybrid_canon_pct': hybrid(canon) / 2,
        'own_long_count': len(own) if own else '',
    })

with open(p('hybrid_splice_prediction.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(r)
