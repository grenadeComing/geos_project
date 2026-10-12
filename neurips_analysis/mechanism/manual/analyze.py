import json,glob,os,re,subprocess,sys,shutil
R='/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github/'
C=R+'result/custom/'
BIN='/Users/kezhang/Desktop/projects/geos_PHREEQC/phreeqc-3.8.6-17100/src/phreeqc'
DB=R+'archive_acm_2026/all_materials/phreeqc.dat'
HERE=os.path.dirname(os.path.abspath(__file__))
V=json.load(open(R+'human_evaluation/verification.json'))
Q={q['id']:q for q in V['questions']}
DS={}
for l in open(R+'datasets/neurips_dataset_200_clean.jsonl'):
    r=json.loads(l); DS[r['id']]=r
NUM=re.compile(r'(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')
def gold_spec(gs):
    gs=gs.strip()
    m=re.match(r'^([-+]?)(\d*)\.?(\d*)(?:[eE]([-+]?\d+))?$',gs)
    if not m: return None
    dec=len(m.group(3)); exp=int(m.group(4)) if m.group(4) else 0
    g=float(gs); tol=0.5*10**(-dec+exp)*1.0001
    return g,tol
def matches_in(text,gs,absval=False):
    sp=gold_spec(gs)
    if sp is None: return []
    g,tol=sp; out=[]
    for m in NUM.finditer(text):
        try: t=float(m.group())
        except: continue
        if abs(t-g)<=tol or (absval and abs(abs(t)-abs(g))<=tol):
            out.append(m.start())
    return out
def events(run,folder):
    lg=sorted(glob.glob(C+run+'/work_space/'+folder+'/logs/chat_*.jsonl'))[-1]
    ev=[json.loads(l) for l in open(lg)]
    return ev,lg
def block_context(lines):
    ctx=[]; cur='-'
    for ln in lines:
        s=ln.strip()
        m=re.match(r'^(Reaction step \d+|Initial solution \d+|Transport step\s+\d+.*|Using solution \d+.*|Beginning of batch-reaction calculations|Beginning of initial solution calculations|Beginning of transport calculations|Cell\s+\d+.*|Reading input data for simulation \d+)',s)
        if m: cur=m.group(1)[:60]
        ctx.append(cur)
    return ctx
def rerun(run,folder,ev,tag):
    base=os.path.join(HERE,'reruns',tag)
    if os.path.exists(base): shutil.rmtree(base)
    os.makedirs(base)
    files={}; execs=[]
    for e in ev:
        if e['action']!='tool': continue
        a=e['args'] if isinstance(e['args'],dict) else {}
        if e['tool']=='write_file':
            files[a.get('path','').lstrip('./')]=a.get('new_content','')
        if e['tool']=='execute_phreeqc':
            k=len(execs); d=os.path.join(base,'exec_%d'%k); os.makedirs(d)
            for p,c in files.items():
                fp=os.path.join(d,p); os.makedirs(os.path.dirname(fp),exist_ok=True); open(fp,'w').write(c)
            before=set(os.listdir(d))
            inp=a.get('input_path','').lstrip('./')
            r=e['result'] if isinstance(e['result'],dict) else {}
            if os.path.exists(os.path.join(d,inp)):
                subprocess.run([BIN,inp,'result.out',DB],cwd=d,capture_output=True,text=True,timeout=600)
            outs={}
            for f in os.listdir(d):
                if f not in before or f=='result.out':
                    try: outs[f]=open(os.path.join(d,f),errors='replace').read()
                    except: pass
            execs.append(dict(step=e.get('step'),ok=r.get('ok'),logged_sizes=r.get('file_sizes'),outs=outs,input=inp))
    return execs
def analyze(x,show_traj=True,L=1200,rerun_on=True):
    run=x['run']; folder=x['folder']; item=x['item']
    q=Q[item]; gs=q['gold_value']
    print('='*100)
    print('ITEM',item,x['model'],x['protocol'],run,'pred',x.get('prediction'),'correct',x.get('is_correct'))
    print('QUANTITY:',q['quantity'],'| GOLD',q['gold'],gs,'| computed',q['computed'][:150])
    qtxt=DS[item]['question']; print('Q:',qtxt[-700:].replace('\r',''))
    ev,lg=events(run,folder)
    steps=sum(1 for e in ev if e['action']=='assistant')
    cnt={}
    for e in ev:
        if e['action']=='tool': cnt[e['tool']]=cnt.get(e['tool'],0)+1
    end=[e for e in ev if e['action']=='run_end']
    print('STEPS',steps,'TOOLS',cnt,'END',end[-1] if end else None)
    absval='dissolved' in q['quantity'] or 'delta' in q['quantity']
    # tool results
    print('--- gold matches in TOOL RESULTS')
    for e in ev:
        if e['action']!='tool': continue
        r=e['result']; s=json.dumps(r) if isinstance(r,dict) else str(r)
        s=s.encode().decode('unicode_escape',errors='ignore') if False else s
        txt=(r.get('content','') or '')+(r.get('head','') or '')+(r.get('tail','') or '')+(r.get('result_out_content','') or '') if isinstance(r,dict) else s
        if isinstance(r,dict) and 'matches' in r: txt+=json.dumps(r['matches']).replace('\\n','\n').replace('\\t','\t')
        if isinstance(r,dict) and r.get('result_out_toc'): txt+=json.dumps(r['result_out_toc'])
        ms=matches_in(txt,gs,absval)
        if ms:
            print(' step',e.get('step'),e['tool'],json.dumps(e['args'])[:120],'n=%d'%len(ms))
            for p in ms[:6]:
                ls=txt.rfind('\n',0,p); le=txt.find('\n',p)
                print('     >',txt[ls+1:le if le>0 else None][:150])
    if rerun_on:
        tag=run[-12:]+'_'+str(item)
        execs=rerun(run,folder,ev,tag)
        print('--- gold matches in RERUN OUTPUTS (%d execs)'%len(execs))
        for k,ex in enumerate(execs):
            print(' exec',k,'step',ex['step'],'ok',ex['ok'],'logged',ex['logged_sizes'],'rerun sizes',{f:len(t) for f,t in ex['outs'].items()})
            for f,t in ex['outs'].items():
                lines=t.split('\n'); ctx=block_context(lines) if f=='result.out' else None
                hits=[]
                for i,ln in enumerate(lines):
                    if matches_in(ln,gs,absval): hits.append(i)
                if hits:
                    print('   ',f,'hits',len(hits))
                    for i in hits[:12]:
                        print('      L%d [%s] %s'%(i+1,ctx[i] if ctx else '',lines[i][:140]))
                    if len(hits)>12: print('      ... last: L%d [%s] %s'%(hits[-1]+1,ctx[hits[-1]] if ctx else '',lines[hits[-1]][:140]))
    if show_traj:
        print('--- TRAJECTORY')
        for e in ev:
            a=e['action']
            if a=='assistant':
                c=e.get('content','') or ''
                tc=e.get('tool_calls')
                if isinstance(tc,str):
                    try: tc=eval(tc)
                    except Exception: tc=[tc]
                print('\n#A%s | %s'%(e.get('step'),c[:2500]))
                for t in (tc or []):
                    if isinstance(t,dict): print('  CALL',t.get('name'),str(t.get('arguments'))[:1500])
            elif a=='tool':
                r=e['result']
                if isinstance(r,dict):
                    r=dict(r)
                    if 'stderr' in r: r['stderr']=r['stderr'][-200:]
                    if 'content' in r and e['tool']=='read_file' and 'database_reference' in json.dumps(e['args']): r['content']=r['content'][:100]+'...'
                    s=json.dumps(r)
                else: s=str(r)
                print('  RESULT(%d) %s'%(len(s),s[:L]))
if __name__=='__main__':
    S=json.load(open(os.path.join(HERE,'samples.json')))
    model,proto,item=sys.argv[1],sys.argv[2],int(sys.argv[3])
    L=int(sys.argv[4]) if len(sys.argv)>4 else 1200
    for x in S:
        if x['model']==model and x['protocol']==proto and x['item']==item:
            analyze(x,L=L); break
