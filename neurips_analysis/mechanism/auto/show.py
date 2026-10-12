import json,glob,sys,os
R='/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github/result/custom'
def show(run, ws_no, maxlen=1500):
    f=sorted(glob.glob(f'{R}/{run}/work_space/dataset_q_{ws_no:05d}/logs/*.jsonl'))[-1]
    for l in open(f):
        e=json.loads(l)
        a=e['action']
        if a=='user': print('USER', e['content'][-400:].replace('\n',' | '))
        elif a=='assistant':
            print('ASSIST',e['step'],(e.get('content') or '')[:maxlen].replace('\n',' | '), [ (t['name'], t['arguments'][:maxlen]) for t in e.get('tool_calls',[])])
        elif a=='tool':
            r=e['result']
            if e['tool']=='read_file' or e['tool']=='grep_file':
                print('  RESULT', json.dumps(r)[:maxlen])
            elif e['tool']=='execute_phreeqc':
                rr={k:v for k,v in r.items() if k not in ('stderr','stdout')}
                print('  EXEC', json.dumps(rr)[:maxlen])
        else: print(a, json.dumps(e)[:200])
if __name__=='__main__':
    show(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv)>3 else 1500)
