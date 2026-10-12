import json,glob
R='/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github/result/custom/'
def counts(run,folder):
    lg=sorted(glob.glob(R+run+'/work_space/'+folder+'/logs/*.jsonl'))[-1]
    ev=[json.loads(l) for l in open(lg)]
    c=dict(steps=0,n_execute=0,n_execute_ok=0,n_read=0,n_read_output=0,n_grep=0,status=None)
    for e in ev:
        if e['action']=='assistant': c['steps']+=1
        if e['action']=='run_end': c['status']=e.get('status')
        if e['action']=='tool':
            t=e['tool']; a=json.dumps(e['args'])
            if t=='execute_phreeqc':
                c['n_execute']+=1
                if isinstance(e['result'],dict) and e['result'].get('ok'): c['n_execute_ok']+=1
            if t=='read_file':
                c['n_read']+=1
                if 'database_reference' not in a: c['n_read_output']+=1
            if t=='grep_file': c['n_grep']+=1
    return c
if __name__=='__main__':
    for fn in ['samples.json','gain_samples.json']:
        for x in json.load(open(fn)):
            print(fn[:4],x['model'],x['protocol'],x['item'],counts(x['run'],x['folder']))
