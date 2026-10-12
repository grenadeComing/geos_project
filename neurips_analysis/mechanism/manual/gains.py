import json,sys,io,contextlib
sys.argv=['x']
exec(open('sample.py').read().split("print('verified'")[0])
import analyze
out=[]
for m in ['gpt51','gpt52']:
    t=toc(m)
    gains=sorted(i for i in ver if t[i][1]['is_correct'] and not raw[m][i]['is_correct'])
    print(m,'gains',gains)
    k=len(gains)/6; p=[gains[int(j*k)] for j in range(6)]; print(' picked',p)
    for i in p:
        run='rebuttal_mcq200_raw100k_rerun_%s_20260723'%m
        x=dict(model=m,protocol='Raw',run=run,item=i,folder='dataset_q_%05d'%i,prediction=raw[m][i]['prediction'],is_correct=raw[m][i]['is_correct'],group='raw_loss')
        out.append(x)
        x2=dict(model=m,protocol='TOC',run=t[i][0],item=i,folder='dataset_q_%05d'%(t[i][1]['row_index']+1),prediction=t[i][1]['prediction'],is_correct=t[i][1]['is_correct'],group='raw_loss_pair')
        out.append(x2)
        for xx in (x,x2):
            buf=io.StringIO()
            with contextlib.redirect_stdout(buf):
                try: analyze.analyze(xx,L=700)
                except Exception as e: print('ERR',repr(e))
            open('dumps/gain_%s_%s_%03d.txt'%(m,xx['protocol'],i),'w').write(buf.getvalue())
json.dump(out,open('gain_samples.json','w'),indent=1)
