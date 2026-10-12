import json,csv
from counts import counts
S=json.load(open('samples.json')); G=json.load(open('gain_samples.json'))
# (model,protocol,item) -> (stage, covered, committed, raw_input_vs_pair, confidence, summary)
L={
# ---------------- GPT-5.1 TOC losses ----------------
('gpt51','TOC',18):('BUILD','n','n','','high',"Wrote '-steps 10 in 100' (10 s, not 1000 s) and 'pH 2 charge'; .sel shows pe -2.61 at step 100 (ref -0.081 never produced). Read only its .sel, then wrote '<<< None >>>' (abstained)."),
('gpt51','TOC',26):('SEEN_NOT_USED','y','n','','high',"Same wrong time/H2S build as Raw, but Si speciation count is build-independent. Read its own .sel (m_H4SiO4, m_H3SiO4-, m_H2SiO4-2 nonzero; 3 'not found' columns = 0) through step 100, then wrote '<<< 2 >>>' (a value, not a letter; 2 = option D). Never read result.out."),
('gpt51','TOC',36):('NOT_SEEN','n','y','','high',"Build produced log IAP -12.26 (result.out L1040, 'Reaction step 10'). TOC lists repeated 'Distribution of species/Saturation indices' entries without step labels; agent read L920-971 = the step-9 block (log IAP -11.67, read ends at 'Reaction step 10.'), matched no option, answered A."),
('gpt51','TOC',51):('NOT_SEEN','n','y','','medium',"Asked for free-ion molalities in SELECTED_OUTPUT (m_Fe+2 1.13e-08 > m_Fe+3 2.0e-10) and answered Fe(2). The redox totals Fe(2)=1.153e-08 vs Fe(3)=4.323e-04 (result.out L196/L208) were never read. Borderline: total Fe 4.3234e-04 was visible in the .sel, so Fe(3) dominance was inferable."),
('gpt51','TOC',79):('BUILD','n','y','','high',"Added 'EQUILIBRIUM_PHASES N2(g) 0 0', forcing SI(N2)=0. Read the right SI block (N2(g) 0.00), no option matched, answered D (guess)."),
('gpt51','TOC',89):('BUILD','n','y','','high',"'pH 7 charge' (reference uses fixed pH 7) -> pe -2.151, never 3.508. Read .sel only, answered B (guess)."),
('gpt51','TOC',121):('SEEN_NOT_USED','y','y','','high',"Correct build. Read .sel pH 2.75065 at step 1 (printed result.out value: 2.751) and picked A 2.750 (truncated instead of rounded). Never read result.out."),
('gpt51','TOC',127):('SEEN_NOT_USED','y','y','','high',"Correct build. -high_precision .sel gave mu 4.173705615675e-02. Picked distractor A '4.1741e-02' instead of C 4.174e-02 (printed result.out value). Never read result.out."),
('gpt51','TOC',149):('BUILD','n','n','','high',"Inflow solution mis-numbered (SOLUTION 0 = Ca/Cl, NaCl as SOLUTION 100, never used), so NaX is never formed. Spent 12 reads scanning blind line ranges (4000-4200, 8200-8400, 240-340, ...) and hit the 24-step cap with no answer."),
('gpt51','TOC',175):('BUILD','n','n','','high',"'pH 7 charge' + 'S(6) 15 as SO4' -> mu 4.405e-04 at step 5 (ref 3.385e-04). Read .sel and wrote '<<< None of the provided options (A-D) match ... 4.404813484191e-04. >>>'"),
('gpt51','TOC',187):('BUILD','n','y','','high',"'C(-4) 5e-3' with default mmol/kgw = 1000x too little carbon -> pe -4.329. Read the correct block, no option matched, answered D (guess)."),
('gpt51','TOC',194):('BUILD','n','y','','high',"REACTION 'H2O -1; 1.0 moles in 50 steps' (removed 1 mol, not 50) plus 'Alkalinity 150 as HCO3'; ref 0.90895 never produced. VISCOUS punched 0. Also got an empty TOC for a 406k output: not in the cap45 splice list. Answered B (guess)."),
# ---------------- GPT-5.2 TOC losses ----------------
('gpt52','TOC',12):('BUILD','n','n','','high',"Never got 1000 s / 100 steps right ('-steps 10 100', '10 in 100', '10 1000'), then tuned parameters (parms 1e4) to chase options. 8 executes, 6 .sel reads, hit the 24-step cap with no answer. Ref -9.260e-03 never produced."),
('gpt52','TOC',24):('BUILD','n','y','','high',"H2S(g) 1 atm (not 5); TOT('S(-2)') at step 100 = 4.07e-02 (ref 3.041e-02 gas delta never produced). Read the .sel tail, answered D (guess)."),
('gpt52','TOC',31):('NOT_SEEN','n','y','','high',"First build ('pH 7 charge', same as Raw) produced I=4.066e-04 at 100 C (result.out L548), but USER_PUNCH IONIC_STRENGTH punched zeros. Agent never read result.out of that run. It 'fixed' the input by also removing 'charge', read the new blocks (I=3.158e-02), then: 'Ionic strength = 3.158e-02 ... Among the choices, this corresponds to D. 4.064e-04'."),
('gpt52','TOC',46):('SEEN_NOT_USED','y','y','','high',"Read the Ca block (L175-215: 6 Ca species) and stated 'That is 6 aqueous Ca species, so the correct option is: <<< D >>>', but D is 4 (A is 6). Letter-mapping slip."),
('gpt52','TOC',57):('BUILD','n','n','','high',"'pH 5' without 'charge' (reference input uses pH 5 charge) -> pH 5.786, never 7.417. Wrote '<<< None of the above >>>'."),
('gpt52','TOC',63):('SEEN_NOT_USED','y','y','','high',"Read L145-212, which includes the C(4) block listing 13 species (count is the same despite its 'pH 5' build). Answered D (4), apparently counting only free carbonate species. Raw counted the complexes."),
('gpt52','TOC',94):('SEEN_NOT_USED','y','y','','high',"Same build as Raw. .sel si_Quartz -2.2052, picked A -2.20 (truncation). result.out prints -2.21; never read result.out."),
('gpt52','TOC',113):('BUILD','n','y','','high',"Used 'units mg/L' for Cl (reference uses default mmol/kgw) -> a(H+) 1.7934e-07. Rationalized: 'This corresponds to option A (1.850e-07)'."),
('gpt52','TOC',132):('BUILD','n','y','','high',"'pH 7 charge' -> CaHSO4+ 3.1741e-09 (ref 3.162e-09 with fixed pH 7). Picked 'Closest option: C. 3.168e-09'. Raw had the same wrong value and guessed A."),
('gpt52','TOC',147):('BUILD','n','y','','high',"'units mol/kgw' made Na/Ca/Cl 1 mol/kg -> SC ~8.4e4 uS/cm, never 123. Punched SC to .sel and answered A (guess)."),
('gpt52','TOC',154):('BUILD','n','n','','medium',"Infill solution defined as SOLUTION 11 (unused); SOLUTION 0 = Ca/Cl, so cell 1 never turns NaCl. 0.89026 occurs only as solution 11's own viscosity, never read. 7 executes, mostly .sel reads, hit the step cap with no answer."),
('gpt52','TOC',175):('BUILD','n','y','','high',"'pH 7 charge' -> mu 4.405e-04 at step 5 (ref 3.385e-04). Answered D (guess)."),
# ---------------- GPT-5.2 Grep losses ----------------
('gpt52','Grep',12):('BUILD','n','y','','high',"'-parms 1.0 1.0 10.0' -> alkalinity -9.216e-03 at step 100 (ref -9.260e-03). Grep + read of .sel tail, answered A (guess)."),
('gpt52','Grep',39):('BUILD','n','y','','high',"Gas phase -pressure 16 (sum of partials) and minerals 0 10 (reference 0 0) -> pe -3.216. Iterated on gas variants (5 executes, 3 greps), answered D (guess)."),
('gpt52','Grep',60):('NOT_SEEN','n','y','','medium',"Same input as Raw. Its USER_PUNCH EQUI() returns FINAL moles in the assemblage. Agent read that as 'moles reacted (positive = dissolved): Talc 23.333 ... Sepiolite 0' and answered Talc. Never grepped the 'Phase assemblage' Delta column (Sepiolite -10)."),
('gpt52','Grep',95):('BUILD','n','y','','high',"'pH 7 charge' -> MgOH+ 7.778e-06 (ref 6.816e-06). 'Among the choices, the closest is D. 6.817e-06'. Raw had the same wrong value and rationalized C."),
('gpt52','Grep',121):('SEEN_NOT_USED','y','y','','high',"Same build as Raw. Read .sel 2.75065 and wrote 'pH = 2.75065, which rounds to 2.750' -> A. Printed value is 2.751."),
('gpt52','Grep',161):('SEEN_NOT_USED','y','y','','medium',"Build produced a(Na+)=9.444e-04 (cell 3, shift 10). Agent grepped its .sel of la_Na+ (-3.024864024343) and converted: '= 9.443e-04' (true 9.4436e-04 -> 9.444e-04). Seen in log form, mis-converted/truncated."),
# ---------------- Sonnet TOC failures ----------------
('sonnet46','TOC',9):('BUILD','n','y','','high',"INCREMENTAL_REACTIONS true + 'Cu' (not Cu(2)); tried 6 rate variants. Best step-100 pH 2.12972 (printed 2.130); the gold 2.129 appears only at step 99 of run 1. Reasoned 'rounds to 2.130 (option B)' but wrote D. 24 steps."),
('sonnet46','TOC',10):('BUILD','n','y','','high',"Same INCREMENTAL_REACTIONS build: SI Witherite at step 100 = -9.352 (-9.36 appears at steps 98/99). Read result.out at step 100 ('-9.35') and answered A."),
('sonnet46','TOC',18):('BUILD','n','y','','high',"Tried EQUILIBRIUM_PHASES H2S, GAS_PHASE at 5 atm, '10 in 100' vs '1000 in 100'. pe stayed -0.12 to -0.17, ref -0.081 never produced. Hit the step cap but had written C."),
('sonnet46','TOC',21):('BUILD','n','y','','high',"Gas phase -pressure 5 (reference: 1 atm, H2S 5) -> I=2.16545e-02 (printed 2.165e-02). Picked D 2.164e-02 rather than the nearer B."),
('sonnet46','TOC',39):('BUILD','n','y','','high',"Minerals 'Calcite/Gypsum/Aragonite 0 10' (reference 0 0) -> pe -4.243. Explored gas-phase readings ('partial pressures summing to 16 atm') over 6 runs, never changed the mineral amounts, answered C (guess)."),
('sonnet46','TOC',47):('BUILD','n','y','','high',"'pH 3' without 'charge' and 'units mol/L' (reference: pH 3 charge) -> pH 5.502. Read the correct blocks repeatedly, answered D (guess)."),
('sonnet46','TOC',65):('BUILD','n','y','','high',"'pH 5' without 'charge' -> SI Sepiolite(d) -8.81 (ref -8.82). Read the SI block correctly and answered A (-8.81), faithful to its own output."),
('sonnet46','TOC',148):('BUILD','n','y','','high',"'units mmol/L' (reference: default mmol/kgw) -> Na at cell 1, shift 5 = 9.872e-04 (ref 9.871e-04). Read .sel, answered C, faithful to its own output."),
}
# Raw pairs: stage OK = value produced+seen+used; BUILD = correct answer without the value ever being produced
R={
('gpt51',18):('BUILD','n','y','both_wrong','high',"Also wrong ('pH 2 charge', -volume 1): inline tail shows pe -2.682 at step 100, but agent answered D anyway (lucky guess, no reasoning)."),
('gpt51',26):('OK','y','y','same_build','high',"Same wrong build (10 s, H2S 1 atm). Inline step-100 'Si' block lists H4SiO4, H3SiO4-, H2SiO4-2 -> A."),
('gpt51',36):('OK','y','y','same_build','high',"Equivalent input. Full 48k output inline incl. step-10 SI block (Calcite log IAP -12.26) -> D."),
('gpt51',51):('OK','y','y','different_query','high',"Same chemistry, but asked for totals Fe(2)/Fe(3). SELECTED_OUTPUT came after END so the .sel is empty. Inline 'Fe(2) 1.153e-08 / Fe(3) 4.323e-04' -> B."),
('gpt51',79):('OK','y','y','different_build','high',"No N2(g) equilibrium phase, so the value was produced: inline 'N2(g) 2.09' -> C."),
('gpt51',89):('BUILD','n','y','both_wrong','high',"Same wrong 'pH 7 charge' build: inline pe -2.151, answered D anyway (lucky guess)."),
('gpt51',121):('OK','y','y','same_build','high',"Same build; inline 'pH = 2.751' (printed precision matches the key) -> B, no .sel read."),
('gpt51',127):('OK','y','y','same_build','high',"Same build; inline 'Ionic strength = 4.174e-02' -> C."),
('gpt51',149):('BUILD','n','y','both_wrong','high',"Also wrong (SOLUTION 0 = Ca/Cl, inflow SOLUTION 11 unused, redefined exchange log_k 0). No NaX value; answered B (lucky guess)."),
('gpt51',175):('OK','y','y','different_build','high',"Built without 'charge'/'as SO4' first time -> inline 3.385e-04 -> B."),
('gpt51',187):('OK','y','y','different_build','high',"Wrote 'C(-4) 5 mmol/kgw' first time -> inline pe -4.584 -> C."),
('gpt51',194):('OK','y','y','different_build','high',"'H2O -1; 50.0 moles' -> inline viscosity 0.90895 -> D."),
('gpt52',12):('BUILD','n','y','both_wrong','high',"Also 10 s ('-steps 10 in 100') -> alkalinity -1.239e-02, no option matches. Answered B anyway (lucky guess)."),
('gpt52',24):('BUILD','n','y','both_wrong','high',"Also wrong (10 s): delta 1.520e-01. 'Expressed in the option format (~3.04e-02 as listed), the correct choice given is: <<< A >>>' (rationalized lucky guess)."),
('gpt52',31):('OK','y','y','same_build','high',"Same first build and the same zero-punch bug, but the inline output showed 'Ionic strength = 4.066e-04' at 100 C -> C in 6 steps."),
('gpt52',46):('OK','y','y','same_build','high',"Same chemistry; struggled with counting BASIC, then punched the 6 species -> A. The species block was inline throughout."),
('gpt52',57):('OK','y','y','different_build','high',"'pH 5 charge' first time -> inline pH 7.417 -> A."),
('gpt52',63):('OK','y','y','different_build','high',"'pH 5 charge'; punch counter returned 0, but agent enumerated the 13 C(4) species from the inline block -> B."),
('gpt52',94):('OK','y','y','same_build','high',"Same build; inline 'Quartz -2.21' -> C (the .sel -2.2052 was not consulted)."),
('gpt52',113):('OK','y','y','different_build','high',"Default units -> inline/sel a(H+) 1.851e-07 -> C."),
('gpt52',132):('BUILD','n','y','both_wrong','high',"Same wrong 'pH 7 charge' build: .sel/inline CaHSO4+ 3.174e-09. Answered A (the key) with no stated reasoning; lucky guess (TOC picked the nearer C)."),
('gpt52',147):('OK','y','y','different_build','high',"Default units -> inline 'Specific Conductance = 123' -> D."),
('gpt52',154):('BUILD','n','y','both_wrong','medium',"All 7 executes failed ('Solution 2 is needed for transport'), so transport never ran. The only option-matching number inline was the NaCl infill solution's initial viscosity 0.89026 -> C (proxy that happens to equal the cell-1/shift-6 value)."),
('gpt52',175):('OK','y','y','different_build','high',"No 'charge' -> inline 3.385e-04 at step 5 -> B."),
('gpt52',39):('OK','y','y','different_build','high',"Gas phase -pressure 1 and minerals 0 0 first time -> inline pe -4.646 -> A."),
('gpt52',60):('OK','y','y','same_build','high',"Identical input incl. the same EQUI() punch, but the inline 'Phase assemblage' Delta column (Sepiolite -1.000e+01) was in context -> B."),
('gpt52',95):('BUILD','n','y','both_wrong','high',"Same wrong 'pH 7 charge' build: MgOH+ 7.7782e-06. 'Matching to the provided options (rounded): 6.816e-06 -> C' (false rationalization, lucky)."),
('gpt52',121):('OK','y','y','same_build','high',"Same build; inline 'pH = 2.751' -> B (no .sel read)."),
('gpt52',161):('OK','y','y','different_build','medium',"Different but valid build; inline transport-step block shows Na+ activity 9.444e-04 -> A."),
}
rows=[]
for x in S:
    c=counts(x['run'],x['folder'])
    m,p,i=x['model'],x['protocol'],x['item']
    if p=='Raw':
        pair='Grep' if x['group']=='grep_loss_pair' else 'TOC'
        st,cov,com,inp,conf,summ=R[(m,i)]
        pg='%s-vs-Raw pair'%pair
    else:
        st,cov,com,inp,conf,summ=L[(m,p,i)]; pg=x['group']
    alt = 'NO_COMMIT' if com=='n' else st
    rows.append(dict(model=m,protocol=p,run=x['run'],item_id=i,pair_group=pg,stage=st,stage_if_nocommit_first=alt,
        outcome=('correct' if x.get('is_correct') else ('no_commit' if x.get('prediction') is None else 'wrong')),
        prediction=x.get('prediction'),steps=c['steps'],n_execute=c['n_execute'],n_execute_ok=c['n_execute_ok'],n_read=c['n_read'],
        n_read_output=c['n_read_output'],n_grep=c['n_grep'],hit_step_cap=('y' if c['status']=='max_steps_reached' else 'n'),
        covered_value_range=cov,committed=com,raw_input_vs_pair=inp,confidence=conf,summary=summ))
cols=list(rows[0].keys())
with open('labels.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
print(len(rows),'rows written')
import collections
for key in [('gpt51','TOC'),('gpt52','TOC'),('gpt52','Grep'),('sonnet46','TOC')]:
    print(key, collections.Counter(r['stage'] for r in rows if (r['model'],r['protocol'])==key))
print('Raw', collections.Counter((r['model'],r['pair_group'],r['stage']) for r in rows if r['protocol']=='Raw'))
