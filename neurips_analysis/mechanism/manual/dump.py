import json,sys,glob,os
R='/Users/kezhang/Desktop/projects/geos_PHREEQC/geos_github/result/custom/'
run=sys.argv[1]; item=int(sys.argv[2]); L=int(sys.argv[3]) if len(sys.argv)>3 else 1500
W=R+run+'/work_space/dataset_q_%05d/'%item
logs=sorted(glob.glob(W+'logs/chat_*.jsonl'))
print('LOGS',logs)
for lg in logs[-1:]:
  for l in open(lg):
    e=json.loads(l)
    a=e['action']
    if a=='run_start': print('#START',e)
    elif a=='user': print('#USER\n',e['content'][:3000])
    elif a=='assistant':
        print('\n#A step',e.get('step'),'|',e.get('content','')[:3000])
        tc=e.get('tool_calls')
        if isinstance(tc,str):
            try: tc=eval(tc)
            except Exception: pass
        if tc:
            for t in tc:
                args=t.get('arguments') if isinstance(t,dict) else t
                print('  CALL',t.get('name') if isinstance(t,dict) else '',str(args)[:2500])
    elif a=='tool':
        r=e['result']
        if isinstance(r,dict):
            r=dict(r)
            if 'stderr' in r: r['stderr']=r['stderr'][-300:]
            s=json.dumps(r)
        else: s=str(r)
        print('  RESULT(%d chars)'%len(s), s[:L])
    else: print('#',a,e)
