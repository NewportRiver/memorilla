"""LITE-008L ordinary token-ID textual decoder competence evaluation."""
from __future__ import annotations
import json,time,resource
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008l.ledger import phase_start,phase_end,record
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008L'

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.monotonic();phase_start('textual_competence',seed=20261124,mode='ordinary_token_ids')
 manifest=json.loads((ROOT/'textual-dataset-manifest.json').read_text()); rows=manifest['rows']; byid={r['id']:r for r in rows};tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=False);model.requires_grad_(False);model.eval();model.config.use_cache=False
 forward_count=0
 def relation(r): return 'capital' if r['kind']=='familiar' else 'color'
 def query(r): return f'What is the {relation(r)} of {r["entity"]}?'
 def evidence(entity,value,r): return f'Memory: {entity} has {relation(r)} {value}.'
 def prompt(r,fmt,condition):
  ev=None
  if condition=='relevant':ev=evidence(r['entity'],r['value'],r)
  elif condition=='counterfactual':ev=evidence(r['entity'],r['alternate'],r)
  elif condition=='unrelated':
   other=next(x for x in rows if x['entity']!=r['entity']);ev=evidence(other['entity'],r['value'],r)
  elif condition=='conflicting':ev=evidence(r['entity'],r['alternate'],r)+' '+evidence(r['entity'],r['value'],r)
  q=query(r)
  if fmt=='question_answer':return (ev+' ' if ev else '')+q+' Answer:'
  if fmt=='evidence_question':return (ev+'\n' if ev else '')+q+' Answer:'
  return (ev+' ' if ev else '')+f"The {relation(r)} of {r['entity']} is"
 def score_prompt(prompt_text,answer,alternate):
  nonlocal forward_count
  p=tok(prompt_text,add_special_tokens=True)['input_ids'];a=manifest['tokenizations'][answer];b=manifest['tokenizations'][alternate]
  if len(a)==1 and len(b)==1:
   full=torch.tensor([p])
   with torch.no_grad(): logits=model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False).logits.float()
   forward_count+=1;vals=logits[:,-1,:].log_softmax(-1)[0];la=float(vals[a[0]]);lb=float(vals[b[0]]);return la,lb,float(vals.argmax()),a,b
  full=torch.tensor([p+a]);full_b=torch.tensor([p+b])
  with torch.no_grad(): logits=model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False).logits.float()
  with torch.no_grad(): logits_b=model(input_ids=full_b,attention_mask=torch.ones_like(full_b),use_cache=False).logits.float()
  forward_count+=2;startpos=len(p)-1
  def total(x,ids): return x[:,startpos:startpos+len(ids),:].log_softmax(-1).gather(-1,torch.tensor(ids).view(1,-1,1)).sum()
  la=float(total(logits,a));lb=float(total(logits_b,b));return la,lb,float(logits[:,-1,:].argmax(-1)[0]),a,b
 def case(r,fmt,condition):
  a,b,g,ta,tb=score_prompt(prompt(r,fmt,condition),r['value'],r['alternate']);ma=a/len(ta);mb=b/len(tb);d={'id':r['id'],'format':fmt,'condition':condition,'a_logprob':a,'b_logprob':b,'a_minus_b_margin':a-b,'a_mean_logprob':ma,'b_mean_logprob':mb,'answer_tokens':ta,'alternate_tokens':tb,'greedy_first_token':int(g)};record('textual_competence','case_completed',case_id=f'{r["id"]}:{fmt}:{condition}',forward_count=forward_count);return d
 conditions=['query_only','relevant','counterfactual','unrelated','conflicting'];cal={};
 for rid in manifest['partitions']['calibration']:
  r=byid[rid]
  for fmt in manifest['prompt_formats']:
   for c in conditions:cal[f'{rid}:{fmt}:{c}']=case(r,fmt,c)
 def acc(data,c,reverse=False):
  x=[v for v in data.values() if v['condition']==c];return sum(v['a_minus_b_margin']<0 if reverse else v['a_minus_b_margin']>0 for v in x)/len(x)
 fmt_scores={}
 for fmt in manifest['prompt_formats']:
  vals={c:acc({k:v for k,v in cal.items() if v['format']==fmt},c,c=='counterfactual') for c in conditions};vals['score']=vals['relevant']+vals['counterfactual']+vals['unrelated'];fmt_scores[fmt]=vals
 selected=max(manifest['prompt_formats'],key=lambda f:fmt_scores[f]['score']);dev={}
 for rid in manifest['partitions']['development']:
  r=byid[rid]
  for c in conditions:dev[f'{rid}:{selected}:{c}']=case(r,selected,c)
 summary={'schema':'LITE-008L-textual-results-v1','selected_format':selected,'calibration_format_scores':fmt_scores,'calibration':{c:acc(cal,c,c=='counterfactual') for c in conditions},'development':{c:acc(dev,c,c=='counterfactual') for c in conditions},'development_count':len(manifest['partitions']['development']),'calibration_count':len(manifest['partitions']['calibration']),'raw_calibration':cal,'raw_development':dev,'forward_count':forward_count,'wall_seconds':time.monotonic()-start,'decoder_parameters_frozen':all(not p.requires_grad for p in model.parameters()),'protected_final_holdout_accessed':False};(ROOT/'textual-competence-results.json').write_text(json.dumps(summary,indent=2)+'\n');phase_end('textual_competence',forward_count=forward_count,wall_seconds=summary['wall_seconds'],selected_format=selected);print(json.dumps({'selected_format':selected,'calibration':summary['calibration'],'development':summary['development'],'forward_count':forward_count,'seconds':summary['wall_seconds']}))
if __name__=='__main__':main()
