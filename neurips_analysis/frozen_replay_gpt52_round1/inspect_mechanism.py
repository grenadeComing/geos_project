"""Read-only analysis of round-one frozen replay; writes only this analysis folder."""
import collections
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / 'result/custom/frozen_replay_gpt52'
spec = importlib.util.spec_from_file_location('matcher', REPO / 'neurips_analysis/mechanism/auto/analyze.py')
matcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matcher)
items = json.loads((REPO / 'neurips_analysis/mechanism/auto/items.json').read_text())
manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text())


def match_piece(text, path, specs):
    main = Path(path).name == 'result.out'
    clean = matcher.clean_text(text, 'out' if main else 'plain')
    kinds = collections.Counter()
    found = matcher.match_text(clean, specs, kinds=kinds, hp=not main)
    traps = matcher.round_trunc_pairs(clean, specs, hp=not main)
    return found, kinds, traps


def branch(cp, item_id, condition, specs, written):
    folder = ROOT / f'runs/repeat_01/item_{item_id:05d}' / condition
    result = json.loads((folder / 'result.json').read_text())
    state = json.loads((folder / 'state.json').read_text())
    assert state['messages'][1:len(cp['messages']) + 1] == cp['messages']
    boundary = len(cp['messages']) + 1
    assert state['messages'][boundary]['tool_call_id'] == cp['execute_tool_call_id']
    pieces = []
    if condition == 'full':
        pieces.append(('result.out', cp['payload_full'].get('result_out_content', ''), 'inline'))
    reads = []
    failures = []
    denied = []
    for event in state['events']:
        name, args, output = event['tool'], event.get('args', {}), event.get('result', {})
        path = args.get('path', '')
        if output.get('ok') is False:
            failures.append(event)
        if name not in {'read_file', 'list_file', 'write_file'} or (name == 'write_file' and path != 'final_answer.txt'):
            denied.append(event)
        if name == 'read_file':
            kind = ('report' if Path(path).name == 'result.out' else
                    'reference' if Path(path).name == 'database_reference.txt' else
                    'input' if path in written else 'other_output')
            read = dict(path=path, kind=kind, ok=output.get('ok'), start_line=args.get('start_line'),
                        end_line=args.get('end_line'), showing=output.get('showing'),
                        chars=len(output.get('content', '')), error=output.get('error'))
            reads.append(read)
            if output.get('ok') and kind in {'report', 'other_output'}:
                pieces.append((path, output.get('content', ''), 'read'))
    found = collections.Counter(); kinds = collections.Counter(); traps = collections.Counter()
    seen_sources = []
    for path, text, source in pieces:
        f, k, t = match_piece(text, path, specs)
        found.update(f); kinds.update(k); traps.update(t)
        if f[cp['item']['answer']]:
            seen_sources.append(dict(path=path, source=source, gold_kinds=[kind for (letter, kind) in k if letter == cp['item']['answer']]))
    gold = cp['item']['answer']
    return {**{key: result[key] for key in ['prediction', 'truth', 'is_correct', 'status', 'calls']},
            'reads': reads, 'read_failures': failures, 'denied_mutation_or_tool_attempts': denied,
            'report_read': any(r['kind'] == 'report' and r['ok'] for r in reads),
            'own_output_read': any(r['kind'] == 'other_output' and r['ok'] for r in reads),
            'gold_seen_after_checkpoint': bool(found[gold]),
            'gold_seen_exact_after_checkpoint': bool(kinds[gold, 'exact']),
            'gold_seen_sources': seen_sources,
            'gold_round_to_pred_trap_seen': bool(traps[gold, result['prediction']]),
            'all_round_trunc_pairs_seen': {a + '->' + b: v for (a,b),v in traps.items()},
            'post_assistant_text': [m.get('content', '') for m in state['messages'][boundary + 1:] if m['role'] == 'assistant' and m.get('content')]}


rows = []
for entry in manifest['eligible']:
    cp_path = ROOT / 'checkpoints' / entry['checkpoint']
    cp = json.loads(cp_path.read_text()); iid = cp['item_id']; gold = cp['item']['answer']
    meta = items[str(iid)]
    specs = matcher.make_spec(matcher.parse_options(cp['item']['question']), meta.get('absmode', False))
    written = set()
    call_map = {}
    for message in cp['messages']:
        for call in message.get('tool_calls', []):
            call_map[call['id']] = call['function']
            if call['function']['name'] == 'write_file':
                written.add(json.loads(call['function']['arguments']).get('path', ''))
    snapshot = cp_path.parent / cp['snapshot']
    produced = []
    for path in sorted(snapshot.rglob('*')):
        rel = str(path.relative_to(snapshot))
        if not path.is_file() or rel in written or path.name in {'database_reference.txt', 'phreeqc.log'}:
            continue
        f,k,t = match_piece(path.read_text(), rel, specs)
        if f[gold]:
            produced.append(dict(path=rel, matches=f[gold], kinds=[kind for (letter,kind) in k if letter == gold]))
    prefix_seen = []
    for message in cp['messages']:
        if message['role'] != 'tool': continue
        function = call_map[message['tool_call_id']]
        output = json.loads(message['content']); args = json.loads(function['arguments'])
        if function['name'] == 'execute_phreeqc' and output.get('result_out_content'):
            path, text = 'result.out', output['result_out_content']
        elif function['name'] == 'read_file' and args.get('path') not in written and Path(args.get('path','')).name != 'database_reference.txt':
            path, text = args.get('path',''), output.get('content','')
        else: continue
        f,k,t = match_piece(text,path,specs)
        if f[gold]: prefix_seen.append(dict(path=path, source=function['name']))
    row = dict(item_id=iid, quantity=meta['quantity'], gold=gold, options=meta['options'],
               inline_truncated=cp['report']['inline_truncated'], report_chars=cp['report']['chars'],
               gold_matches_in_snapshot=produced,
               gold_in_checkpoint_report=any(p['path'] == cp['report']['path'] for p in produced),
               gold_in_common_prefix=prefix_seen,
               full=branch(cp,iid,'full',specs,written), toc=branch(cp,iid,'toc',specs,written))
    row['group'] = ('both_correct' if row['full']['is_correct'] and row['toc']['is_correct'] else
                    'full_only' if row['full']['is_correct'] else 'toc_only' if row['toc']['is_correct'] else 'both_wrong')
    rows.append(row)

summary = {'groups': dict(collections.Counter(r['group'] for r in rows))}
for condition in ['full','toc']:
    summary[condition] = dict(n=len(rows), correct=sum(r[condition]['is_correct'] for r in rows),
        status=dict(collections.Counter(r[condition]['status'] for r in rows)),
        report_read=sum(r[condition]['report_read'] for r in rows),
        own_output_read=sum(r[condition]['own_output_read'] for r in rows),
        read_failure_trajectories=sum(bool(r[condition]['read_failures']) for r in rows),
        read_failure_events=sum(len(r[condition]['read_failures']) for r in rows),
        denied_mutation_trajectories=sum(bool(r[condition]['denied_mutation_or_tool_attempts']) for r in rows))
for group in ['full_only','toc_only']:
    rs=[r for r in rows if r['group']==group]
    summary[group] = dict(n=len(rs), ids=[r['item_id'] for r in rs],
        toc_report_read=sum(r['toc']['report_read'] for r in rs),
        toc_own_output_read=sum(r['toc']['own_output_read'] for r in rs),
        toc_report_skip_own_read=sum(not r['toc']['report_read'] and r['toc']['own_output_read'] for r in rs),
        gold_in_report=sum(r['gold_in_checkpoint_report'] for r in rs),
        gold_in_any_snapshot=sum(bool(r['gold_matches_in_snapshot']) for r in rs),
        toc_gold_seen=sum(r['toc']['gold_seen_after_checkpoint'] for r in rs),
        toc_truncation_trap=sum(r['toc']['gold_round_to_pred_trap_seen'] for r in rs),
        toc_errors=[r['item_id'] for r in rs if r['toc']['read_failures']],
        truncated=[r['item_id'] for r in rs if r['inline_truncated']])
(HERE/'item_analysis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(HERE/'discordant_pairs.json').write_text(json.dumps([r for r in rows if r['group'] in {'full_only','toc_only'}],ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
