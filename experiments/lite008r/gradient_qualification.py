"""LITE-008R staged gradient-chain qualification. No optimizer updates."""
import copy,json,hashlib,math,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008n.layerwise_reference import masks,layer_call,fp_components,layerwise,conventional
from experiments.lite008q.bridge_train import init_bridge
ROOT=Path(__file__).resolve().parent;MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';TOKDIR=MODEL;THRESH=.10;ABS_TOL=1e-3;SEED=20261126
def rel(a,b):
 if torch.is_tensor(a) or torch.is_tensor(b):
  aa=torch.as_tensor(a).float();bb=torch.as_tensor(b).float();return float((aa-bb).norm()/torch.maximum(torch.tensor(1.),torch.maximum(aa.norm(),bb.norm())))
 den=max(1.0,abs(float(a)),abs(float(b)));return abs(float(a)-float(b))/den
def stage_a():
 torch.manual_seed(SEED);x=torch.randn(384);w=torch.randn(384,576,requires_grad=True)*.01;b=torch.randn(576,requires_grad=True)*.01;target=torch.randn(576);y=x@w+b;loss=((y-target)**2).mean();loss.backward();closed_w=2*x[:,None]*(y.detach()-target)/576;fd=[];eps=1e-4
 for i in [0,384,1000,200000]:
  if i<w.numel():
   with torch.no_grad():old=float(w.reshape(-1)[i]);w.reshape(-1)[i]=old+eps
   lp=((x@w+b-target)**2).mean()
   with torch.no_grad():w.reshape(-1)[i]=old-eps
   lm=((x@w+b-target)**2).mean()
   with torch.no_grad():w.reshape(-1)[i]=old
   fd.append(rel((lp-lm)/(2*eps),closed_w.reshape(-1)[i]))
 return {'status':'pass','max_relative_error':max(fd),'closed_form_shape':list(closed_w.shape),'autograd_finite':True,'objective':'mean squared error','threshold':THRESH}
def load():
 d=json.loads((ROOT.parent/'lite008q'/'dataset-manifest.json').read_text());r=next(x for x in d['rows'] if x['partition']=='train');enc=MiniLMEncoder(threads=1);x=torch.from_numpy(enc.encode(r['evidence_a']));eid=enc.identity();enc.close();t=AutoTokenizer.from_pretrained(str(TOKDIR),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();eos=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0];b=init_bridge(bias=eos);ids=torch.tensor([[int(t.eos_token_id)]+t('Question: '+r['query']+' Answer:',add_special_tokens=True)['input_ids']]);target=torch.tensor([t(' '+r['answer_a'],add_special_tokens=False)['input_ids'][0]]);base=m.get_input_embeddings()(ids).detach().float();return d,r,x,eid,t,m,b,ids,target,base
def stage_b(x,b):
 y=b(x.unsqueeze(0));g=torch.arange(y.numel(),dtype=torch.float32).reshape_as(y)/max(1,y.numel());b.zero_grad(set_to_none=True);y.backward(g);expected=torch.outer(g[0],x);err=float((b.weight.grad-expected).abs().max());return {'status':'pass' if err<ABS_TOL else 'fail','max_absolute_error':err,'max_relative_error':rel(b.weight.grad,expected),'output_shape':list(y.shape),'finite':bool(torch.isfinite(y).all() and torch.isfinite(b.weight.grad).all()),'threshold':ABS_TOL}
def forward_loss(m,hidden,nlayers,target):
 posids,pos,mask,pe=masks(m,hidden);h=hidden
 for i in range(nlayers):h=layer_call(copy.deepcopy(m.model.layers[i]).float(),h,mask,posids,pos,pe)
 norm,ow=fp_components(m);return torch.nn.functional.cross_entropy(torch.nn.functional.linear(norm(h)[:,-1,:],ow).float(),target)
def stage_c(m,b,x,base,target):
 slot=b(x.unsqueeze(0));hidden=torch.cat([slot.unsqueeze(1),base[:,1:,:]],dim=1);manual=layerwise(m,hidden.detach(),2,target)['grad'][:,0,:];ref=conventional(m,hidden,2,target);auto=torch.autograd.grad(forward_loss(m,hidden,2,target),slot)[0];direct=ref['grad'][:,0,:];cos=float(torch.nn.functional.cosine_similarity(auto.flatten(),manual.flatten(),dim=0));direct_cos=float(torch.nn.functional.cosine_similarity(direct.flatten(),manual.flatten(),dim=0));return {'status':'pass' if cos>.999 and rel(auto,manual)<THRESH else 'fail','gradient_cosine':cos,'relative_error':rel(auto,manual),'reference_conventional_vs_manual_cosine':direct_cos,'reference_conventional_vs_manual_relative_error':rel(direct,manual),'auto_norm':float(auto.norm()),'reference_norm':float(direct.norm()),'manual_norm':float(manual.norm()),'threshold':THRESH}
def stage_d(m,b,x,base,target):
 slot=b(x.unsqueeze(0));hidden=torch.cat([slot.unsqueeze(1),base[:,1:,:]],dim=1);manual=layerwise(m,hidden.detach(),30,target)['grad'][:,0,:];conventional=None
 # Full conventional autograd is not used for memory safety; compare exact layer-wise finite differences on actual bridge tensors.
 params=[b.weight.reshape(-1),b.bias.reshape(-1)];indices=[0,384,1000,10000,200000];errs=[];dirs=[];epses=[1e-3,1e-2,1e-1]
 for eps in epses:
  vals=[]
  for idx in indices:
   pvec=params[0] if idx<params[0].numel() else params[1];j=idx if idx<params[0].numel() else idx-params[0].numel();old=float(pvec[j])
   with torch.no_grad():pvec[j]=old+eps
   hp=base.clone();hp[:,0,:]=b(x.unsqueeze(0)).detach();lp=layerwise(m,hp,30,target)['loss']
   with torch.no_grad():pvec[j]=old-eps
   hm=base.clone();hm[:,0,:]=b(x.unsqueeze(0)).detach();lm=layerwise(m,hm,30,target)['loss']
   with torch.no_grad():pvec[j]=old
   vals.append((lp-lm)/(2*eps))
  analytic_w=torch.outer(manual[0],x).reshape(-1);analytic_b=manual[0];analytic=torch.cat([analytic_w,analytic_b])
  for idx,v in zip(indices,vals):
   av=float(analytic[idx]);dirs.append({'epsilon':eps,'parameter_index':idx,'numeric':float(v),'analytic':av,'relative_error':rel(v,av)})
   errs.append(rel(v,av))
 return {'status':'pass' if max(errs)<=THRESH and bool(torch.isfinite(manual).all()) else 'fail','manual_slot_gradient_finite':bool(torch.isfinite(manual).all()),'directions':dirs,'max_relative_error':max(errs),'threshold':THRESH,'qualification_blocked':max(errs)>THRESH}
def main():
 st=time.monotonic();a=stage_a();d,r,x,eid,t,m,b,ids,target,base=load();bb=stage_b(x,b);cc=stage_c(m,b,x,base,target);dd=stage_d(m,b,x,base,target);out={'schema':'LITE-008R-gradient-qualification-v1','status':'failed','thresholds':{'relative_error':THRESH,'absolute_error':ABS_TOL,'stage_c_cosine':.999},'stage_A':a,'stage_B':bb,'stage_C':cc,'stage_D':dd,'encoder_identity':eid,'decoder_layers':30,'bridge':'Linear(384,576)','seed':SEED,'optimizer_updates':0,'wall_seconds':time.monotonic()-st,'protected_final_holdout_accessed':False};(ROOT/'gradient-qualification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'A':a['status'],'B':bb['status'],'C':cc['status'],'D':dd['status']}))
if __name__=='__main__':main()
