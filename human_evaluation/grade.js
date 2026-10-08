#!/usr/bin/env node
'use strict';
// Organizer only. Reads the original key on this computer; never embeds it in the participant HTML.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
if(process.argv.length!==3){console.error('Usage: node human_evaluation/grade.js /path/to/final-response.json');process.exit(1);}
const manifest=JSON.parse(fs.readFileSync(path.join(__dirname,'selection_manifest.json'),'utf8'));
const bytes=fs.readFileSync(path.join(__dirname,'../datasets',manifest.source_file));
assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),manifest.source_sha256,'Source dataset changed');
const key=new Map(bytes.toString('utf8').trim().split(/\r?\n/).map(JSON.parse).map(r=>[r.id,r.answer]));
const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
assert.equal(input.schema_version,2,'Unsupported response file version');assert.equal(input.kind,'final','Only grade finalized records');
for(const k of ['study_id','selection_sha256','source_sha256','database_sha256'])assert.equal(input[k],manifest[k],'Version mismatch: '+k);
assert(typeof input.participant_code==='string'&&input.participant_code.trim(),'Missing participant code');
assert(typeof input.finalized_at==='string'&&Number.isFinite(Date.parse(input.finalized_at)),'Missing finalization timestamp');
const N=manifest.question_count;
assert(Array.isArray(input.responses)&&input.responses.length===N,'Expected '+N+' response records');
const answers=new Map(input.responses.map(r=>[r.question_id,r]));
assert.equal(answers.size,N,'Duplicate IDs');
const by_scenario=manifest.scenarios.map(s=>{
  let correct=0,idk=0;const n=s.selected_ids.length;
  for(const id of s.selected_ids){
    const r=answers.get(id);
    assert(r&&r.scenario_id===s.id,'Missing or misassigned question '+id);
    assert(['A','B','C','D','E'].includes(r.choice),'Invalid / blank response '+id);
    if(r.choice==='E')idk++;
    if(r.choice===key.get(id))correct++;
  }
  return {scenario_id:s.id,total:n,correct,idk,accuracy:correct/n};
});
const problems=input.responses.filter(r=>r.problem_reported===true).map(r=>({question_id:r.question_id,scenario_id:r.scenario_id,reason:r.problem_reason,note:r.note,choice:r.choice}))
  .sort((a,b)=>a.question_id-b.question_id);
const correct=by_scenario.reduce((n,s)=>n+s.correct,0);
const idk=by_scenario.reduce((n,s)=>n+s.idk,0);
console.log(JSON.stringify({participant_code:input.participant_code,study_id:input.study_id,
  total:N,correct,incorrect_including_idk:N-correct,idk,accuracy:correct/N,
  by_scenario,problem_reports:problems,notes:'E is counted as incorrect and reported separately. Problem reports do not change the score. No independence assumption or confidence interval is imposed by this script.'},null,2));
