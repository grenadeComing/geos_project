import json,csv
from counts import counts
G=json.load(open('gain_samples.json'))
C={
('gpt51','Raw',9):('BUILD','n','y',"'pH 2 charge' (failed to converge), then a variant; ref pH 2.129 never produced. Answered B."),
('gpt51','TOC',9):('BUILD','n','y',"Correct answer, but no run produced 2.129 (4 of 5 executes failed). Lucky A."),
('gpt51','Raw',17):('BUILD','n','y',"H2S(g) 1 atm, 'pH 2 charge'; ref 2.293e-04 never produced. Answered B."),
('gpt51','TOC',17):('BUILD','n','y',"Correct answer, but ref 2.293e-04 not in any output. Lucky A."),
('gpt51','Raw',47):('SEEN_NOT_USED','y','y',"Inline result.out showed 'pH = 5.509', but agent read its high-precision .sel (5.508648) and answered D 5.508. The .sel-precision trap also occurs under Raw."),
('gpt51','TOC',47):('OK','y','y',"Read result.out L110-144 'pH = 5.509' -> B."),
('gpt51','Raw',72):('SEEN_NOT_USED','y','y',"Inline speciation lists 10 Fe(3) species. Answered D (7) with no reasoning."),
('gpt51','TOC',72):('NOT_SEEN','n','y',"'PRINT -species true -reset false' plus a hand-picked .sel list, so the full count was never visible. Answered A (10) correctly without support."),
('gpt51','Raw',116):('BUILD','n','y',"Ref pH 1.013 never produced. Answered B."),
('gpt51','TOC',116):('OK','y','y',".sel 1.01296 -> A."),
('gpt51','Raw',174):('BUILD','n','y',"SC 77 never produced. Answered B."),
('gpt51','TOC',174):('OK','y','y',"Read result.out L320-440: SC 77 -> C."),
('gpt52','Raw',13):('BUILD','n','y',"Ref pH 2.013 at step 10 never produced (681k output). Answered B."),
('gpt52','TOC',13):('OK','y','y',"After 5 executes, .sel 2.01298 at step 10 -> C."),
('gpt52','Raw',17):('BUILD','n','y',"Ref 2.293e-04 never produced. Answered C."),
('gpt52','TOC',17):('BUILD','n','y',"Computed dissolved = 3.3181e-05, then 'This corresponds to option A (2.293e-04) per the provided choices'. Rationalized lucky guess under TOC."),
('gpt52','Raw',53):('SEEN_NOT_USED','y','y',"Inline block has 8 Al species; agent listed 7 (dropped AlHSO4+2 at 3e-23), then 'the intended count corresponds to 6 (option B)'."),
('gpt52','TOC',53):('OK','y','y',"Read the species block: '8 aqueous Al species' -> C."),
('gpt52','Raw',76):('BUILD','n','y',"GAS_PHASE O2(g) -0.7 (a negative partial pressure, so effectively no O2) -> Fe(2) > Fe(3). Answered A, consistent with its own output."),
('gpt52','TOC',76):('OK','y','y',"Read totals Fe(2) 2.028e-10 vs Fe(3) 3.830e-02 -> B."),
('gpt52','Raw',85):('BUILD','n','y',"All 8 executes failed; nothing produced. Answered C."),
('gpt52','TOC',85):('OK','y','y',"Read gas-phase block: CH4 delta 5.277e-02 -> B."),
('gpt52','Raw',155):('BUILD','n','y',"Different transport build; ref 5.506e-06 not in its (untruncated, 50k) output. Answered D."),
('gpt52','TOC',155):('OK','y','y',".sel 5.506245e-06 -> A."),
}
rows=[]
for x in G:
    c=counts(x['run'],x['folder'])
    st,cov,com,summ=C[(x['model'],x['protocol'],x['item'])]
    rows.append(dict(model=x['model'],protocol=x['protocol'],run=x['run'],item_id=x['item'],pair_group='Raw-loss(TOC-gain) pair',stage=st,
      outcome=('correct' if x.get('is_correct') else ('no_commit' if x.get('prediction') is None else 'wrong')),prediction=x.get('prediction'),
      steps=c['steps'],n_execute=c['n_execute'],n_read=c['n_read'],n_read_output=c['n_read_output'],n_grep=c['n_grep'],covered_value_range=cov,committed=com,summary=summ))
with open('labels_rawloss_contrast.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import collections
print(collections.Counter((r['model'],r['protocol'],r['stage']) for r in rows))
