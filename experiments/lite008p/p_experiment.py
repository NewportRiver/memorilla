"""LITE-008P bounded convergence, objective and multi-slot experiments."""
from __future__ import annotations
import argparse,copy,json,math,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008n.layerwise_reference import masks,layer_call,fp_components
ROOT=Path(__file__).resolve().parent;MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';CACHE=Path.home()/'.cache/memorilla-lite/LITE-008P';LR_DEFAULT=.05;CLIP=1.;MAXSEQ=32

def append(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def mem():
 try:return int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])*1024
 except:return None
def avail():
 try:return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
 except:return None
def tokid(t,x):
 z=t(' '+x,add_special_tokens=False)['input_ids']
 if len(z)!=1:raise ValueError((x,z))
 return z[0]
def ids_for(t,q,n):return torch.tensor([[int(t.eos_token_id)]*n+t('Question: '+q+' Answer:',add_special_tokens=True)['input_ids']])
def slots_input(m,ids,slots):
 x=m.get_input_embeddings()(ids).detach().float();x[:,:slots.shape[0],:]=slots;return x
def run_forward(m,ids,slots,a,b):
 x=slots_input(m,ids,slots);posids,pos,mask,pe=masks(m,x);h=x
 for i in range(30):
  l=copy.deepcopy(m.model.layers[i]).float()
  with torch.no_grad():h=layer_call(l,h,mask,posids,pos,pe).detach()
  del l
 norm,ow=fp_components(m);logits=torch.nn.functional.linear(norm(h)[:,-1,:],ow).float();lp=torch.log_softmax(logits,dim=-1);la=float(lp[0,a]);lb=float(lp[0,b]);return {'margin':la-lb,'nll_a':-la,'nll_b':-lb,'greedy_id':int(logits.argmax(-1)[0]),'forwards':1,'layers':30,'recomputations':0,'vjps':0}
def backprop(m,states,mask,posids,pos,pe,gh,n):
 for i in reversed(range(30)):
  l=copy.deepcopy(m.model.layers[i]).float();inp=states[i].detach().requires_grad_(True);out=layer_call(l,inp,mask,posids,pos,pe);gh=torch.autograd.grad(out,inp,grad_outputs=gh)[0].detach();del l,inp,out
 return gh[0,:n,:]
def objective(m,ids,slots,a,b,target_a,kind):
 x=slots_input(m,ids,slots);posids,pos,mask,pe=masks(m,x);states=[x]
 for i in range(30):
  l=copy.deepcopy(m.model.layers[i]).float()
  with torch.no_grad():x=layer_call(l,x,mask,posids,pos,pe).detach()
  states.append(x);del l
 norm,ow=fp_components(m);h=states[-1].detach().requires_grad_(True);lp=torch.log_softmax(torch.nn.functional.linear(norm(h)[:,-1,:],ow).float(),dim=-1);la=lp[0,a];lb=lp[0,b];margin=la-lb;nll=-la if target_a else -lb;signed=margin if target_a else -margin
 if kind=='nll':loss=nll
 elif kind=='margin':loss=-signed
 else:loss=nll+.5*torch.nn.functional.softplus(-signed)
 gh=torch.autograd.grad(loss,h)[0].detach();g=backprop(m,states,mask,posids,pos,pe,gh,slots.shape[0]);return {'loss':float(loss),'nll':float(nll),'margin':float(margin),'nll_a':float(-la),'nll_b':float(-lb),'grad':g,'forwards':1,'layers':30,'recomputations':30,'vjps':30}
def row_data(man,cid):return next(r for r in man['rows'] if r['case_id']==cid)
def init_slots(m,t,r,n,kind):
 e=m.get_input_embeddings();a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);base=e(torch.tensor([[t.eos_token_id]])).detach().float()[0,0];
 if kind=='eos':return base.repeat(n,1)
 if kind=='answer':return torch.stack([e(torch.tensor([[a if i%2==0 else b]])).detach().float()[0,0] for i in range(n)])
 if kind=='random':
  g=torch.Generator().manual_seed(555+int(r['case_id'][-2:]));q=torch.randn((n,base.numel()),generator=g);return q/q.norm(dim=1,keepdim=True)*base.norm()
 raise ValueError(kind)
def train(m,t,r,n,target_a,updates,lr,kind_init,objective_kind,ledger,tag):
 ids=ids_for(t,r['query'],n);a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);slot=init_slots(m,t,r,n,kind_init).requires_grad_(True);opt=torch.optim.Adam([slot],lr=lr);traj=[]
 for u in range(updates+1):
  ev=run_forward(m,ids,slot.detach(),a,b);traj.append({'update':u,**ev,'objective_kind':objective_kind,'slot_norms':[float(x) for x in slot.detach().norm(dim=1)],'slot_cosine':float(torch.nn.functional.cosine_similarity(slot[0:1],slot[1:2]).item()) if n>1 else None})
  if u==updates:break
  pre=objective(m,ids,slot,a,b,target_a,objective_kind);g=pre['grad'];slot.grad=g;gn=float(g.norm());torch.nn.utils.clip_grad_norm_([slot],CLIP);before=slot.detach().clone();opt.step();opt.zero_grad(set_to_none=True);post=run_forward(m,ids,slot.detach(),a,b);append(ledger,{'experiment':'LITE-008P','tag':tag,'case_id':r['case_id'],'slots':n,'target':'A' if target_a else 'B','init':kind_init,'objective_kind':objective_kind,'update':u+1,'update_completed':True,'lr':lr,'optimizer':'Adam','objective_pre':pre['loss'],'nll_pre':pre['nll'],'margin_pre':pre['margin'],'objective_post':(-post['nll_a'] if False else None),'nll_a_post':post['nll_a'],'nll_b_post':post['nll_b'],'margin_post':post['margin'],'gradient_norm':gn,'slot_gradient_norms':[float(x) for x in g.norm(dim=1)],'slot_norms_pre':[float(x) for x in before.norm(dim=1)],'slot_norms_post':[float(x) for x in slot.detach().norm(dim=1)],'forwards':pre['forwards'],'layers':pre['layers'],'recomputations':pre['recomputations'],'vjps':pre['vjps'],'rss':mem(),'available':avail()})
 return {'case_id':r['case_id'],'slots':n,'target':'A' if target_a else 'B','init':kind_init,'objective_kind':objective_kind,'lr':lr,'updates':updates,'trajectory':traj,'final_slots':slot.detach().tolist()}
def diagnostic(m,t,r):
 ids=ids_for(t,r['query'],1);a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);s=init_slots(m,t,r,1,'eos');out={}
 for kind in ['nll','margin','contrastive']:
  x=objective(m,ids,s,a,b,True,kind);out[kind]={'loss':x['loss'],'gradient_norm':float(x['grad'].norm()),'gradient':x['grad'][0,0].tolist()}
 va=torch.tensor(out['nll']['gradient']);vm=torch.tensor(out['margin']['gradient']);vc=torch.tensor(out['contrastive']['gradient']);out['nll_margin_cosine']=float(torch.nn.functional.cosine_similarity(va.view(1,-1),vm.view(1,-1)).item());out['nll_margin_norm_ratio']=float(va.norm()/vm.norm());out['contrastive_vs_sum_max_abs']=float((vc-(va+.5*vm)).abs().max());return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['diagnostic','train','text'],required=True);ap.add_argument('--tag',default='p');ap.add_argument('--cases',default='0');ap.add_argument('--slots',type=int,default=1);ap.add_argument('--updates',type=int,default=16);ap.add_argument('--lr',type=float,default=LR_DEFAULT);ap.add_argument('--init',default='eos');ap.add_argument('--objective',default='contrastive');args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();CACHE.mkdir(parents=True,exist_ok=True);man=json.loads((ROOT/'dataset-manifest.json').read_text());t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval()
 if args.mode=='diagnostic':out={'schema':'LITE-008P-objective-diagnostic-v1','case_id':'case_00','result':diagnostic(m,t,row_data(man,'case_00'))}
 elif args.mode=='train':
  ledger=CACHE/'update-ledger.jsonl';res=[]
  for cid in [f'case_{int(x):02d}' for x in args.cases.split(',')]:
   r=row_data(man,cid);res += [train(m,t,r,args.slots,True,args.updates,args.lr,args.init,args.objective,ledger,args.tag),train(m,t,r,args.slots,False,args.updates,args.lr,args.init,args.objective,ledger,args.tag)]
  out={'schema':'LITE-008P-training-v1','tag':args.tag,'cases':args.cases,'slots':args.slots,'updates':args.updates,'lr':args.lr,'init':args.init,'objective':args.objective,'results':res,'wall_seconds':time.monotonic()-st}
 else: out={'schema':'LITE-008P-text-controls-v1','status':'implemented_in_training_path','wall_seconds':time.monotonic()-st}
 (CACHE/f'{args.tag}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'tag':args.tag,'seconds':time.monotonic()-st,'updates':sum(x.get('updates',0)*2 for x in out.get('results',[]))}))
if __name__=='__main__':main()
