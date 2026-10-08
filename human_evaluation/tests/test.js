#!/usr/bin/env node
'use strict';
const assert=require('assert/strict'),fs=require('fs'),path=require('path'),crypto=require('crypto');
const {pathToFileURL}=require('url');
const {spawnSync}=require('child_process');
const dir=path.resolve(__dirname,'..');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const html=fs.readFileSync(path.join(dir,'for_Jerry.html'),'utf8');
const data=JSON.parse(html.match(/<script id="questionnaire-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const manifest=JSON.parse(fs.readFileSync(path.join(dir,'selection_manifest.json'),'utf8'));
const raw=fs.readFileSync(path.join(dir,'../datasets/neurips_dataset_200_clean.jsonl'));
const verification=JSON.parse(fs.readFileSync(path.join(dir,'verification.json'),'utf8'));
const vstatus=new Map(verification.questions.map(q=>[q.id,q.status]));
const source=new Map(raw.toString('utf8').trim().split(/\r?\n/).map(JSON.parse).map(r=>[r.id,r]));
const questions=data.scenarios.flatMap(s=>s.items);
const N=manifest.question_count,NS=manifest.scenario_count;
assert.equal(NS,20-manifest.held_scenarios.length);assert.equal(N,NS*5);
assert.equal(data.scenarios.length,NS);assert.equal(questions.length,N);
assert.equal(new Set(questions.map(q=>q.id)).size,N);
const heldIds=new Set(manifest.held_scenarios.map(x=>x.id));
assert(data.scenarios.every(s=>!heldIds.has(s.id)),'Held scenario in worksheet');
assert.equal(sha(raw),manifest.source_sha256);
assert.equal(sha(fs.readFileSync(path.join(dir,'phreeqc.dat'))),manifest.database_sha256);
for(const s of data.scenarios){
  assert.equal(s.items.length,5);
  assert.deepEqual(s.items.map(q=>q.id),manifest.scenarios.find(x=>x.id===s.id).selected_ids);
  for(const q of s.items){
    assert(!manifest.excluded_ids.includes(q.id));
    assert.equal(vstatus.get(q.id),'verified','Unverified item selected: Q'+q.id);
    assert(q.target.trim());
    assert.equal(q.source_question,source.get(q.id).question,'Original wording changed');
    assert.deepEqual(Object.keys(q.options),['A','B','C','D']);
    for(const code of ['A','B','C','D']){
      const match=source.get(q.id).question.replace(/\r\n/g,'\n').match(new RegExp('^'+code+'\\.[ \\t]*(.*)$','m'));
      assert.equal(q.options[code],match[1],'Original option changed');
    }
  }
}
function noKey(x){
  if(!x||typeof x!=='object')return;
  for(const [k,v] of Object.entries(x)){
    assert(!['answer','gold','truth','answer_value','answer_tolerance','correct_option','is_correct','author','computed','status'].includes(k),'Answer-key leakage');
    noKey(v);
  }
}
noKey(data);
assert(!/<script[^>]+src=|<link[^>]+href=|fetch\s*\(|XMLHttpRequest|WebSocket/.test(html),'Unexpected external dependency');
const s13=data.scenarios.find(s=>s.id==='S13');
const has125=Boolean(s13&&s13.items.some(q=>q.id===125));
if(has125)assert.notEqual(s13.items.find(q=>q.id===125).preamble,s13.preamble);
assert.equal(vstatus.size,200);
assert.deepEqual(manifest.excluded_ids,[...vstatus].filter(([,s])=>s!=='verified').map(([id])=>id).sort((a,b)=>a-b));
assert(!/\p{Script=Han}/u.test(fs.readFileSync(path.join(dir,'README.md'),'utf8')+html),'Non-English text in participant materials');
const outputs=['for_Jerry.html','selection_manifest.json','author_review.html','for_Sachit.html','phreeqc.dat'];
const before=outputs.map(f=>sha(fs.readFileSync(path.join(dir,f))));
const build=spawnSync(process.execPath,[path.join(dir,'build.js')],{encoding:'utf8'});
assert.equal(build.status,0,build.stderr);
const report=spawnSync(process.execPath,[path.join(dir,'author_report.js')],{encoding:'utf8'});
assert.equal(report.status,0,report.stderr);
const recheckBuild=spawnSync(process.execPath,[path.join(dir,'author_recheck.js')],{encoding:'utf8'});
assert.equal(recheckBuild.status,0,recheckBuild.stderr);
assert.deepEqual(outputs.map(f=>sha(fs.readFileSync(path.join(dir,f)))),before,'Build not deterministic');
console.log('PASS: '+NS+'×5 unique, held scenarios excluded, verified-only pool, fixed sampling, unchanged source wording, no answer keys, English text, deterministic build.');

(async()=>{
  let chromium;
  try{({chromium}=require('playwright'));}catch(_){
    throw new Error('Playwright is needed for UI tests. Set NODE_PATH to an installed Playwright node_modules directory.');
  }
  const out=fs.mkdtempSync(path.join(require('os').tmpdir(),'phreeqc-human-test-'));
  const browser=await chromium.launch({headless:true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE} : {})});
  const errors=[],net=[];
  async function context(viewport={width:1280,height:1000}){
    const c=await browser.newContext({viewport,acceptDownloads:true});
    c.on('page',p=>{
      p.on('pageerror',e=>errors.push(e.message));
      p.on('request',r=>{if(/^https?:/.test(r.url())&&/(for_Jerry|for_Sachit)\.html/.test(p.url()))net.push(r.url());});
    });
    return c;
  }
  const url=pathToFileURL(path.join(dir,'for_Jerry.html')).href;
  const c=await context(),page=await c.newPage();
  await page.goto(url);await page.waitForSelector('#q-'+questions[0].id+'-A');
  assert.equal(await page.locator('.question').count(),5);
  assert.equal(await page.locator('input[type=radio]').count(),25);
  await page.click('#finish');
  assert((await page.textContent('#message')).includes('participant code'));
  await page.fill('#participant','TEST_P01');
  await page.fill('#experience','Synthetic automated test — not a human result');
  const first=questions[0].id;
  await page.check('#q-'+first+'-E');
  await page.locator('#question-'+first+' details summary').click();
  await page.fill('#note-'+first,'Test note: <img src=x onerror=alert(1)>');
  await page.check('#problem-'+first);
  assert(await page.isVisible('#reason-'+first));
  await page.selectOption('#reuse','shared_run');
  await page.fill('#simulation-notes','S01/test.pqi; S01/test.out — synthetic QA only');
  await page.reload();
  assert(await page.isChecked('#q-'+first+'-E'));
  assert.equal(await page.inputValue('#note-'+first),'Test note: <img src=x onerror=alert(1)>');
  assert.equal(await page.inputValue('#reuse'),'shared_run');
  assert(await page.isChecked('#problem-'+first));
  assert.equal(await page.textContent('#total-progress'),'1 / '+N+' answered');
  const [backup]=await Promise.all([page.waitForEvent('download'),page.click('#backup')]);
  const backupPath=path.join(out,'draft.json');await backup.saveAs(backupPath);
  const backupData=JSON.parse(fs.readFileSync(backupPath,'utf8'));
  assert.equal(backupData.kind,'draft');assert.equal(backupData.response_count,1);assert.equal(backupData.schema_version,2);
  assert.equal(backupData.problem_count,1);assert.equal(backupData.responses.find(r=>r.question_id===first).problem_reason,'');
  assert.equal(backupData.responses.length,N);

  const rc=await context(),rp=await rc.newPage();await rp.goto(url);
  rp.on('dialog',d=>d.accept());
  await rp.setInputFiles('#restore',backupPath);
  await rp.waitForFunction(()=>document.getElementById('message').textContent.includes('restored'));
  assert(await rp.isChecked('#q-'+first+'-E'));
  assert.equal(await rp.inputValue('#participant'),'TEST_P01');
  const badVersion=structuredClone(backupData);badVersion.selection_sha256='wrong';
  const badPath=path.join(out,'bad-version.json');fs.writeFileSync(badPath,JSON.stringify(badVersion));
  await rp.setInputFiles('#restore',badPath);
  await rp.waitForFunction(()=>document.getElementById('message').textContent.includes('Restore failed'));
  assert(await rp.isChecked('#q-'+first+'-E'));
  const duplicates=structuredClone(backupData);duplicates.responses[1]=duplicates.responses[0];
  const duplicatePath=path.join(out,'duplicate.json');fs.writeFileSync(duplicatePath,JSON.stringify(duplicates));
  await rp.setInputFiles('#restore',duplicatePath);
  await rp.waitForFunction(()=>document.getElementById('message').textContent.includes('Question IDs'));
  const badChoice=structuredClone(backupData);badChoice.responses[0].choice='F';
  const choicePath=path.join(out,'bad-choice.json');fs.writeFileSync(choicePath,JSON.stringify(badChoice));
  await rp.setInputFiles('#restore',choicePath);
  await rp.waitForFunction(()=>document.getElementById('message').textContent.includes('Invalid answer'));
  await rp.click('#finish');
  assert((await rp.textContent('#message')).includes('without a reason'));
  await rp.selectOption('#reason-'+first,'unclear_wording');
  await rp.click('#finish');
  assert((await rp.textContent('#message')).includes((N-1)+' unanswered'));
  console.log('PASS: save/reload, E distinct from blank, problem reports/notes, backup/restore, malformed import rejection, incomplete-submit guard.');

  let ordinal=0;
  for(const s of data.scenarios){
    await page.click('#nav-'+s.id);
    for(const q of s.items){
      await page.check('#q-'+q.id+'-'+['A','B','C','D','E'][ordinal%5]);ordinal++;
    }
    assert.equal(await page.textContent('#scenario-progress'),'5 / 5 answered');
  }
  if(has125){
    await page.click('#nav-S13');
    assert(await page.locator('#question-125 .source-variant').isVisible());
    assert((await page.textContent('#question-125 .source-variant')).includes('to solution 1'));
  }
  assert.equal(await page.textContent('#total-progress'),N+' / '+N+' answered');
  await page.click('#finish');
  assert((await page.textContent('#message')).includes('without a reason'),'Finish must require a problem reason');
  await page.selectOption('#reason-'+first,'no_correct_option');
  await page.click('#finish');
  await page.waitForSelector('#finish-dialog[open]');
  const [finalDownload]=await Promise.all([page.waitForEvent('download'),page.click('#confirm-finish')]);
  const finalPath=path.join(out,'synthetic-final.json');await finalDownload.saveAs(finalPath);
  const result=JSON.parse(fs.readFileSync(finalPath,'utf8'));
  assert.equal(result.kind,'final');assert.equal(result.response_count,N);assert.equal(result.idk_count,N/5);
  assert.equal(result.responses.length,N);
  assert(await page.isDisabled('#participant'));assert(await page.isDisabled('#restore'));
  assert(await page.isDisabled('#q-'+first+'-A'));assert(await page.isDisabled('#problem-'+first));
  await page.reload();assert(await page.isDisabled('#q-'+first+'-E'));
  assert.equal(await page.textContent('#finish'),'Download final JSON');
  const graded=spawnSync(process.execPath,[path.join(dir,'grade.js'),finalPath],{encoding:'utf8'});
  assert.equal(graded.status,0,graded.stderr);
  const score=JSON.parse(graded.stdout);
  assert.equal(score.idk,N/5);assert.equal(score.total,N);
  assert.deepEqual(score.problem_reports.map(p=>[p.question_id,p.reason]),[[first,'no_correct_option']]);
  const expected=result.responses.filter(r=>r.choice===source.get(r.question_id).answer).length;
  assert.equal(score.correct,expected);
  assert.notEqual(spawnSync(process.execPath,[path.join(dir,'grade.js'),backupPath],{encoding:'utf8'}).status,0);
  console.log('PASS: all '+N+' actual radio inputs, S13 variant preamble if selected, final download, persistent UI lock, local grading including E.');

  const vc=await context(),vp=await vc.newPage();await vp.goto(url);
  await vp.evaluate(()=>document.getElementById('instructions').open=false);
  assert(await vp.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await vp.screenshot({path:path.join(out,'desktop.png')});
  await vp.click('#nav-S16');await vp.evaluate(()=>window.scrollTo(0,0));
  await vp.screenshot({path:path.join(out,'transport.png')});
  await vp.goto(pathToFileURL(path.join(dir,'author_review.html')).href);
  assert.equal(await vp.locator('table').first().locator('tbody tr').count(),20);
  assert.equal(await vp.locator('.item').count(),[...vstatus.values()].filter(s=>s!=='verified').length);
  assert(await vp.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Review page overflow');
  await vp.screenshot({path:path.join(out,'review.png')});
  const mc=await context({width:390,height:844}),mp=await mc.newPage();await mp.goto(url);
  await mp.evaluate(()=>{document.getElementById('instructions').open=false;document.getElementById('scenario-navigation').open=false;});
  assert(await mp.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile overflow');
  await mp.screenshot({path:path.join(out,'mobile.png')});
  await mp.setViewportSize({width:320,height:800});
  assert(await mp.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'320px overflow');
  const bc=await context();
  await bc.addInitScript(()=>{
    Storage.prototype.getItem=function(){throw new DOMException('Storage disabled','SecurityError');};
    Storage.prototype.setItem=function(){throw new DOMException('Storage disabled','SecurityError');};
  });
  const bp=await bc.newPage();await bp.goto(url);await bp.check('#q-'+first+'-E');
  assert((await bp.textContent('#save-status')).includes('unavailable'));
  const [fallback]=await Promise.all([bp.waitForEvent('download'),bp.click('#backup')]);
  const fallbackPath=path.join(out,'storage-disabled-draft.json');await fallback.saveAs(fallbackPath);
  assert.equal(JSON.parse(fs.readFileSync(fallbackPath,'utf8')).response_count,1);

  // Author recheck page (grouped by cause)
  const recheckHtml=fs.readFileSync(path.join(dir,'for_Sachit.html'),'utf8');
  const rd=JSON.parse(recheckHtml.match(/<script id="recheck-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const flaggedIds=[...vstatus].filter(([,x])=>x!=='verified').map(([id])=>id).sort((a,b)=>a-b);
  assert.deepEqual(rd.groups.flatMap(g=>g.ids).sort((a,b)=>a-b),flaggedIds,'Every flagged question in exactly one group');
  const setupTotal=rd.groups.flatMap(g=>g.setups).length,total=flaggedIds.length+setupTotal;
  const rc2=await context(),ap=await rc2.newPage();ap.on('dialog',d=>d.accept());
  await ap.goto(pathToFileURL(path.join(dir,'for_Sachit.html')).href);
  assert.equal(await ap.locator('.qblock').count(),flaggedIds.length);
  assert.equal(await ap.locator('.setup-card').count(),setupTotal);
  assert.equal(await ap.textContent('#progress-count'),'0 of '+total+' settled');
  assert.equal(await ap.locator('#q22 .opts li').count(),4,'Each question lists its four options');
  const unstable=rd.groups.find(g=>g.id==='unstable').ids;
  await ap.fill('#reviewer','Test reviewer (synthetic)');
  await ap.check('#group-unstable-drop');
  assert.equal(await ap.textContent('#progress-count'),unstable.length+' of '+total+' settled','Group decision settles every question in it');
  const ov=unstable[0];
  assert(await ap.locator('#decision-'+ov+'-revise').isHidden());
  await ap.check('#override-'+ov);await ap.check('#decision-'+ov+'-revise');
  assert(await ap.isVisible('#stem-'+ov));await ap.fill('#stem-'+ov,'Revised question text (synthetic)');
  await ap.check('#group-wrong-key-change_key');
  assert.equal(await ap.textContent('#q22 .card-head .pill:last-child'),'Needs input');
  await ap.selectOption('#newkey-22','B');
  assert.equal(await ap.textContent('#q22 .card-head .pill:last-child'),'Settled');
  const ms=rd.groups.find(g=>g.id==='missing-stem').ids[0];
  await ap.check('#group-missing-stem-revise');
  assert.equal(await ap.textContent('#q'+ms+' .card-head .pill:last-child'),'Needs input','Missing sentence must be written before it counts');
  await ap.fill('#stem-'+ms,'What is the pH of the solution?');
  await ap.check('#setup-S17-revise_setup');await ap.fill('#setuptext-S17','Setup with units (synthetic)');
  await ap.check('#decision-3-keep');
  const pqi=path.join(out,'s11_author.pqi');fs.writeFileSync(pqi,'SOLUTION 1\n pH 7.5\nEND\n');
  await ap.setInputFiles('#attach-S11',pqi);await ap.waitForSelector('#input-rows .chip');
  const settledNow=unstable.length+1+1+1+1;
  assert.equal(await ap.textContent('#progress-count'),settledNow+' of '+total+' settled');
  await ap.check('#only-open');
  assert(await ap.locator('#group-unstable').isHidden());assert(await ap.locator('#q22').isHidden());
  assert(await ap.locator('#q'+rd.groups.find(g=>g.id==='missing-stem').ids[1]).isVisible());
  const [recheckDl]=await Promise.all([ap.waitForEvent('download'),ap.click('#download')]);
  const recheckPath=path.join(out,'recheck.json');await recheckDl.saveAs(recheckPath);
  const rec=JSON.parse(fs.readFileSync(recheckPath,'utf8'));
  assert.equal(rec.schema,'phreeqc-author-recheck');assert.equal(rec.schema_version,2);assert.equal(rec.questions.length,flaggedIds.length);
  const get=id=>rec.questions.find(x=>x.id===id);
  assert.equal(get(unstable[1]).decision,'drop');assert.equal(get(unstable[1]).decided_by,'group');
  assert.equal(get(ov).decision,'revise');assert.equal(get(ov).decided_by,'question');assert.equal(get(ov).stem,'Revised question text (synthetic)');
  assert.equal(get(22).decision,'change_key');assert.equal(get(22).new_key,'B');
  assert.equal(get(ms).stem,'What is the pH of the solution?');
  assert.equal(rec.scenarios.find(x=>x.id==='S17').decision,'revise_setup');
  assert.deepEqual(rec.inputs.map(f=>[f.scenario_id,f.filename]),[['S11','s11_author.pqi']]);
  assert.equal(rec.settled.questions+rec.settled.setups,settledNow);
  await ap.reload();
  assert(await ap.isChecked('#group-unstable-drop'));assert(await ap.isChecked('#override-'+ov));
  assert.equal(await ap.inputValue('#reviewer'),'Test reviewer (synthetic)');
  const fp=await (await context()).newPage();fp.on('dialog',d=>d.accept());
  await fp.goto(pathToFileURL(path.join(dir,'for_Sachit.html')).href);
  await fp.setInputFiles('#restore',recheckPath);
  await fp.waitForFunction(()=>document.getElementById('status-line').textContent.includes('Restored'));
  assert.equal(await fp.inputValue('#newkey-22'),'B');assert.equal(await fp.textContent('#progress-count'),settledNow+' of '+total+' settled');
  const badRec=structuredClone(rec);badRec.verification_sha256='x';const badRecPath=path.join(out,'bad-recheck.json');fs.writeFileSync(badRecPath,JSON.stringify(badRec));
  await fp.setInputFiles('#restore',badRecPath);
  await fp.waitForFunction(()=>document.getElementById('status-line').textContent.includes('Could not restore'));
  await fp.setViewportSize({width:360,height:800});
  assert(await fp.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Recheck page overflow at 360px');
  await ap.setViewportSize({width:1100,height:1600});await ap.uncheck('#only-open');await ap.evaluate(()=>window.scrollTo(0,0));
  await ap.screenshot({path:path.join(out,'recheck.png')});
  await ap.locator('#group-moles-molality').screenshot({path:path.join(out,'recheck-group.png')});
  console.log('PASS: author recheck page: groups cover all flagged items, group decisions, per-question override, required fields, file attach, JSON download, reload, restore and version check, 360px fit.');
  assert.equal(errors.length,0,errors.join('\n'));assert.equal(net.length,0,'External network requests');
  assert.equal(sha(fs.readFileSync(path.join(dir,'../datasets/neurips_dataset_200_clean.jsonl'))),manifest.source_sha256);
  await browser.close();
  console.log('PASS: desktop/mobile/320px fit, author review page, storage-denied backup, no JS errors, no external requests, source unchanged.');
  console.log('QA screenshots and synthetic responses (NOT human data): '+out);
})().catch(error=>{console.error(error);process.exit(1);});
