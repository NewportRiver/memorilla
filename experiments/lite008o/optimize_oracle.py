"""LITE-008O controlled counterfactual optimization with layer-wise FP32 VJPs."""
from __future__ import annotations
import argparse,copy,json,math,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008n.layerwise_reference import masks,layer_call,fp_components
ROOT=Path(__file__).resolve().parent;CACHE=Path.home()/'.cache/memorilla-lite/LITE-008O';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
STEPS=[0,1,2,4,8];LR=.05;CLIP=1.;MAX_UPDATES=8

def rss():
 try:return int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])*1024
 except:return None
def avail():
 try:return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
 except:return None
def append(path,event):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def make_ids(tok,q):return torch.tensor([[int(tok.eos_token_id)]+tok('Question: '+q+' Answer:',add_special_tokens=True)['input_ids']])
def token(tok,x):
 z=tok(' '+x,add_special_tokens=False)['input_ids']
 if len(z)!=1:raise ValueError(f'not single token {x}: {z}')
 return z[0]
def forward(m,ids,slot,tid_a,tid_b):
 x=m.get_input_embeddings()(ids).detach().float();x[:,0,:]=slot;posids,pos,mask,pe=masks(m,x);h=x
 for i in range(30):
  layer=copy.deepcopy(m.model.layers[i]).float()
  with torch.no_grad():h=layer_call(layer,h,mask,posids,pos,pe).detach()
  del layer
 norm,ow=fp_components(m);logits=torch.nn.functional.linear(norm(h)[:,-1,:],ow).float();lp=torch.log_softmax(logits,dim=-1);la=float(lp[0,tid_a]);lb=float(lp[0,tid_b]);return {'logp_a':la,'logp_b':lb,'margin_a_minus_b':la-lb,'nll_a':-la,'nll_b':-lb,'greedy_id':int(logits.argmax(-1)[0]),'complete_decoder_forwards':1,'layer_executions':30,'backward_vjps':0,'layer_recomputations':0}
def objective(m,ids,slot,tid_a,tid_b,target_is_a,contrastive):
 x=m.get_input_embeddings()(ids).detach().float();x[:,0,:]=slot.detach();posids,pos,mask,pe=masks(m,x);states=[x]
 for i in range(30):
  layer=copy.deepcopy(m.model.layers[i]).float()
  with torch.no_grad():x=layer_call(layer,x,mask,posids,pos,pe).detach()
  states.append(x);del layer
 norm,ow=fp_components(m);h=states[-1].detach().requires_grad_(True);logits=torch.nn.functional.linear(norm(h)[:,-1,:],ow).float();lp=torch.log_softmax(logits,dim=-1);la=lp[0,tid_a];lb=lp[0,tid_b];margin=la-lb;nll=-la if target_is_a else -lb;signed=margin if target_is_a else -margin;loss=nll+(.5*torch.nn.functional.softplus(-signed) if contrastive else 0.)
 gh=torch.autograd.grad(loss,h)[0].detach();recomp=0
 for i in reversed(range(30)):
  layer=copy.deepcopy(m.model.layers[i]).float();inp=states[i].detach().requires_grad_(True);out=layer_call(layer,inp,mask,posids,pos,pe);gh=torch.autograd.grad(out,inp,grad_outputs=gh)[0].detach();recomp+=1;del layer,out,inp
 return {'loss':float(loss.detach()),'nll':float(nll.detach()),'margin_a_minus_b':float(margin.detach()),'nll_a':float(-la.detach()),'nll_b':float(-lb.detach()),'grad':gh[0,0].clone(),'complete_decoder_forwards':1,'layer_executions':30,'backward_vjps':30,'layer_recomputations':30}
def optimize(m,tok,row,target_is_a,contrastive,mode,updates,ledger):
 ids=make_ids(tok,row['query']);a=token(tok,row['answer_a']);b=token(tok,row['answer_b']);base=m.get_input_embeddings()(ids).detach().float()[0,0];slot=base.clone().requires_grad_(True);opt=torch.optim.Adam([slot],lr=LR);traj=[];total={'complete_decoder_forwards':0,'layer_executions':0,'backward_vjps':0,'layer_recomputations':0}
 for step in range(updates+1):
  ev=forward(m,ids,slot.detach(),a,b);traj.append({'update':step,'objective':None,'mode':mode,**{k:v for k,v in ev.items() if k not in ('complete_decoder_forwards','layer_executions','backward_vjps','layer_recomputations')},'embedding_norm':float(slot.detach().norm())});
  if step==updates:break
  before=objective(m,ids,slot,a,b,target_is_a,contrastive);g=before.pop('grad');total['complete_decoder_forwards']+=before.pop('complete_decoder_forwards');total['layer_executions']+=before.pop('layer_executions');total['backward_vjps']+=before.pop('backward_vjps');total['layer_recomputations']+=before.pop('layer_recomputations');slot.grad=g;torch.nn.utils.clip_grad_norm_([slot],CLIP);pre=slot.detach().clone();opt.step();opt.zero_grad(set_to_none=True);after=forward(m,ids,slot.detach(),a,b);append(ledger,{'experiment':'LITE-008O','case_id':row['case_id'],'mode':mode,'target_direction':'A' if target_is_a else 'B','update_attempted':step+1,'update_completed':step+1,'objective_pre_update':before['loss'],'objective_post_update':None,'nll_a_post':after['nll_a'],'nll_b_post':after['nll_b'],'margin_post':after['margin_a_minus_b'],'gradient_norm':float(g.norm()),'embedding_norm_pre':float(pre.norm()),'embedding_norm_post':float(slot.detach().norm()),'complete_decoder_forwards':total['complete_decoder_forwards']+1,'layer_executions':total['layer_executions']+30,'backward_vjps':total['backward_vjps'],'layer_recomputations':total['layer_recomputations'],'rss_bytes':rss(),'available_bytes':avail(),'seed':20261124})
  # objective fields are retained from the pre-update calculation in a side trajectory only
  traj[-1]['objective']=before['loss'] if step<updates else None
 return {'case_id':row['case_id'],'mode':mode,'target_direction':'A' if target_is_a else 'B','updates_completed':updates,'trajectory':traj,'final_embedding':slot.detach().tolist(),'counts':total}
def control(m,tok,row):
 ids=make_ids(tok,row['query']);a=token(tok,row['answer_a']);b=token(tok,row['answer_b']);emb=m.get_input_embeddings();eos=emb(ids).detach().float()[0,0];zero=torch.zeros_like(eos);rng=torch.Generator().manual_seed(7000+int(row['case_id'][-2:]));rand=torch.randn(eos.shape,generator=rng);rand=rand/rand.norm()*eos.norm();ea=emb(torch.tensor([[a]])).detach().float()[0,0];eb=emb(torch.tensor([[b]])).detach().float()[0,0];out={}
 for name,s in [('zero',zero),('scale_matched_random',rand),('genuine_answer_a',ea),('genuine_answer_b',eb),('initial_eos',eos)]:out[name]=forward(m,ids,s,a,b)
 out['random_search']=[]
 for i in range(8):
  q=torch.randn(eos.shape,generator=torch.Generator().manual_seed(8000+i));q=q/q.norm()*eos.norm();out['random_search'].append(forward(m,ids,q,a,b))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['contrastive','nll_only'],required=True);args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.monotonic();CACHE.mkdir(parents=True,exist_ok=True);data=json.loads((ROOT/'dataset-manifest.json').read_text());text=json.loads((ROOT/'textual-prerequisite-results.json').read_text());primary=[r for r in data['rows'] if r['case_id'] in ['case_08','case_09','case_10','case_11']];assert all(r['case_id'] in text['eligible_oracle_cases'] for r in primary);tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();ledger=CACHE/'update-ledger.jsonl';append(ledger,{'experiment':'LITE-008O','event':'started','mode':args.mode,'seed':20261124,'primary_cases':[r['case_id'] for r in primary],'optimizer':'Adam','learning_rate':LR,'gradient_clip_norm':CLIP,'max_updates_per_embedding':MAX_UPDATES,'contrastive_weight':.5});results=[]
 selected=primary if args.mode=='contrastive' else primary[:2]
 for r in selected: results.extend([optimize(m,tok,r,True,args.mode=='contrastive',args.mode,MAX_UPDATES,ledger),optimize(m,tok,r,False,args.mode=='contrastive',args.mode,MAX_UPDATES,ledger)])
 controls={r['case_id']:control(m,tok,r) for r in primary} if args.mode=='contrastive' else {};out={'schema':'LITE-008O-optimization-v1','status':'completed','mode':args.mode,'primary_case_ids':[r['case_id'] for r in selected],'optimizer':'Adam','learning_rate':LR,'gradient_clip_norm':CLIP,'max_updates_per_embedding':MAX_UPDATES,'results':results,'controls':controls,'wall_seconds':time.monotonic()-start,'protected_final_holdout_accessed':False};(CACHE/f'optimization-{args.mode}-results.json').write_text(json.dumps(out,indent=2)+'\n');append(ledger,{'experiment':'LITE-008O','event':'completed','mode':args.mode,'wall_seconds':out['wall_seconds'],'total_optimizer_updates':sum(x['updates_completed'] for x in results)});print(json.dumps({'seconds':out['wall_seconds'],'embeddings':len(results),'updates':sum(x['updates_completed'] for x in results)}))
if __name__=='__main__':main()
