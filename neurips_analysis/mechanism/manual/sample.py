import json
R='/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github'
C=R+'/result/custom/'
v=json.load(open(R+'/human_evaluation/verification.json'))
ver={q['id']:q for q in v['questions'] if q['status']=='verified'}
cap=set(int(x) for x in open(R+'/neurips_analysis/baseline_snapshot/cap_affected_union45_ids.txt').read().split())
def load(run):
    d={}
    for l in open(C+run+'/results.jsonl'):
        r=json.loads(l); d[r['index']]=r
    return d
def toc(model):
    base={'gpt51':'neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt51','gpt52':'neurips200_clean_rerun_20260503_safe_redo_run1_toc_gpt52','sonnet46':'neurips200_clean_rerun_20260504_sonnet_run1_toc_sonnet46'}[model]
    fix='neurips200_tocfix_uncapped_capaffected45_toc_'+model
    b=load(base); f=load(fix)
    out={}
    for i in range(1,201):
        if i in cap: out[i]=(fix,f[i])
        else: out[i]=(base,b[i])
    return out
raw={'gpt51':load('rebuttal_mcq200_raw100k_rerun_gpt51_20260723'),'gpt52':load('rebuttal_mcq200_raw100k_rerun_gpt52_20260723')}
grep52=load('neurips200_grep_gpt52')
print('verified',len(ver), 'cap&ver', len(cap&set(ver)))
samples=[]
def pick(ids,n):
    ids=sorted(ids)
    if len(ids)<=n: return ids
    k=len(ids)/n
    return [ids[int(j*k)] for j in range(n)]
for m in ['gpt51','gpt52']:
    t=toc(m)
    acc_t=sum(1 for i in ver if t[i][1]['is_correct'])
    acc_r=sum(1 for i in ver if raw[m][i]['is_correct'])
    losses=[i for i in ver if not t[i][1]['is_correct'] and raw[m][i]['is_correct']]
    gains=[i for i in ver if t[i][1]['is_correct'] and not raw[m][i]['is_correct']]
    print(m,'toc acc',acc_t,'raw acc',acc_r,'losses',len(losses),sorted(losses),'gains',len(gains))
    p=pick(losses,12); print(' picked',p)
    for i in p:
        samples.append(dict(model=m,protocol='TOC',run=t[i][0],item=i,group='toc_loss'))
        samples.append(dict(model=m,protocol='Raw',run='rebuttal_mcq200_raw100k_rerun_%s_20260723'%m,item=i,group='toc_loss_pair'))
gl=[i for i in ver if not grep52[i]['is_correct'] and raw['gpt52'][i]['is_correct']]
print('grep52 losses',len(gl),sorted(gl))
p=pick(gl,6); print(' picked',p)
for i in p:
    samples.append(dict(model='gpt52',protocol='Grep',run='neurips200_grep_gpt52',item=i,group='grep_loss'))
    samples.append(dict(model='gpt52',protocol='Raw',run='rebuttal_mcq200_raw100k_rerun_gpt52_20260723',item=i,group='grep_loss_pair'))
t=toc('sonnet46')
sf=[i for i in ver if not t[i][1]['is_correct']]
print('sonnet toc fail verified',len(sf),sorted(sf))
p=pick(sf,8); print(' picked',p)
for i in p:
    samples.append(dict(model='sonnet46',protocol='TOC',run=t[i][0],item=i,group='sonnet_fail'))
json.dump(samples,open('samples.json','w'),indent=1)
print(len(samples))
