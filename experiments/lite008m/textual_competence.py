"""Corrected fixed-format ordinary token-ID competence evaluation for LITE-008M."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008l.ledger import record
ROOT=Path.home()/'.cache/memorilla-lite/LITE-008M';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);d=json.loads((ROOT/'balanced-dataset-manifest.json').read_text());rows=d['rows'];by={r['id']:r for r in rows};tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=False);model.requires_grad_(False);model.eval();model.config.use_cache=False;forward=0;record('textual_competence','started',seed=d['seed'],format='question_answer_fixed')
 def prompt(r,c):
  q=f'Question: What is the {r["relation"]} of {r["entity"]}? Answer:'
  ev=''
  if c=='relevant':ev=f'Memory: {r["entity"]} has {r["relation"]} {r["value"]}. '
  elif c=='counterfactual':ev=f'Memory: {r["entity"]} has {r["relation"]} {r["alternate"]}. '
  elif c=='unrelated':
   other=next(x for x in rows if x['id']!=r['id']);ev=f'Memory: {other["entity"]} has {r["relation"]} {r["value"]}. '
  elif c=='conflicting':ev=f'Memory: {r["entity"]} has {r["relation"]} {r["alternate"]}. Memory: {r["entity"]} has {r["relation"]} {r["value"]}. '
  return ev+q
 def score(r,c):
  nonlocal forward
  p=tok(prompt(r,c),add_special_tokens=True)['input_ids'];out={}
  answer_ids={label:tok(' '+ans,add_special_tokens=False)['input_ids'] for label,ans in [('a',r['value']),('b',r['alternate'])]}
  if len(answer_ids['a'])==1 and len(answer_ids['b'])==1:
   with torch.no_grad():logits=model(input_ids=torch.tensor([p]),attention_mask=torch.ones((1,len(p)),dtype=torch.long),use_cache=False).logits.float()
   forward+=1;lp=logits[:,-1,:].log_softmax(-1)[0];out['a']=float(lp[answer_ids['a'][0]]);out['b']=float(lp[answer_ids['b'][0]])
  else:
   for label,ids in answer_ids.items():
    full=torch.tensor([p+ids]);
    with torch.no_grad():logits=model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False).logits.float()
    forward+=1;lp=logits[:,len(p)-1:len(p)-1+len(ids),:].log_softmax(-1);out[label]=float(lp.gather(-1,torch.tensor(ids).view(1,-1,1)).sum())
  z={'id':r['id'],'condition':c,'a_minus_b_margin':out['a']-out['b'],'a_logprob':out['a'],'b_logprob':out['b'],'multi_token':r['multi_token']};record('textual_competence','case_completed',case_id=f'{r["id"]}:{c}',forward_count=forward);return z
 conditions=['query_only','relevant','counterfactual','unrelated','conflicting'];raw=[]
 for r in rows:
  for c in conditions:raw.append(score(r,c))
 def subset(part,multi=None):return [x for x in raw if by[x['id']]['partition']==part and (multi is None or x['multi_token']==multi)]
 def acc(xs,c,rev=False):
  z=[x for x in xs if x['condition']==c];return sum((x['a_minus_b_margin']<0)==rev for x in z)/len(z)
 dev=[x for x in raw if by[x['id']]['partition']=='development'];primary=[x for x in dev if not x['multi_token']];multi=[x for x in dev if x['multi_token']];cal=[x for x in raw if by[x['id']]['partition']=='calibration']
 def metrics(xs):return {c:acc(xs,c,c=='counterfactual') for c in conditions}
 def means(xs):return {c:sum(x['a_minus_b_margin'] for x in xs if x['condition']==c)/len([x for x in xs if x['condition']==c]) for c in conditions}
 result={'schema':'LITE-008M-textual-competence-v1','status':'completed','format':'question_answer_fixed','calibration':metrics(cal),'development_primary_single_token':metrics(primary),'development_multi_token':metrics(multi),'development_all':metrics(dev),'mean_margins_primary':means(primary),'mean_margins_multi':means(multi),'query_only_preference_primary':means(primary)['query_only'],'forward_count':forward,'decoder_parameters_frozen':all(not p.requires_grad for p in model.parameters()),'protected_final_holdout_accessed':False,'raw':raw};(ROOT/'textual-competence-results.json').write_text(json.dumps(result,indent=2)+'\n');record('textual_competence','completed',forward_count=forward,development_pairs=28);print(json.dumps({'primary':result['development_primary_single_token'],'multi':result['development_multi_token'],'forward_count':forward}))
if __name__=='__main__':main()
