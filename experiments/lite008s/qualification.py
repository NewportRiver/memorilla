"""LITE-008S corrected deterministic decoder VJP qualification; zero optimizer updates."""
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008n.layerwise_reference import masks,layer_call,fp_components,layerwise,conventional
from experiments.lite008q.bridge_train import init_bridge
ROOT=Path(__file__).resolve().parent;QROOT=ROOT.parent/'lite008q';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';THRESH=.10;ABS_TOL=1e-3;COS_TOL=.999;SEED=20261126

def metrics(a,b):
 a=a.detach().float();b=b.detach().float();d=a-b;an=float(a.norm());bn=float(b.norm());return {'absolute_max':float(d.abs().max()),'absolute_l2':float(d.norm()),'reference_relative_l2':float(d.norm()/max(an,1e-12)),'symmetric_relative_l2':float(d.norm()/max((an+bn)/2,1e-12)),'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)) if an and bn else 1.0,'gradient_norm_ratio':float(bn/max(an,1e-12))}
def scalar_metrics(a,b):
 aa=float(a);bb=float(b);ab=abs(aa-bb);return {'absolute_error':ab,'reference_relative_error':ab/max(abs(aa),1e-12),'symmetric_relative_error':ab/max((abs(aa)+abs(bb))/2,1e-12)}
def sha(t):return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def forward_only(m,hidden,n,target):
 with torch.no_grad():
  posids,pos,mask,pe=masks(m,hidden.detach().float());x=hidden.detach().float()
  for i in range(n):x=layer_call(copy.deepcopy(m.model.layers[i]).float(),x,mask,posids,pos,pe)
  norm,ow=fp_components(m);logits=torch.nn.functional.linear(norm(x)[:,-1,:],ow).float();return torch.nn.functional.cross_entropy(logits,target),logits
def stage_a():
 torch.manual_seed(SEED);torch.set_default_dtype(torch.float64);x=torch.randn(7);w=torch.randn(7,5,requires_grad=True);b=torch.randn(5,requires_grad=True);target=torch.randn(5);y=x@w+b;loss=((y-target)**2).mean();gw,gb=torch.autograd.grad(loss,(w,b));cw=2*x[:,None]*(y.detach()-target)/5;cb=2*(y.detach()-target)/5;rows=[];eps=1e-5
 for kind,idx in [('w',0),('w',8),('b',0),('b',4)]:
  p=w if kind=='w' else b;flat=p.reshape(-1);old=float(flat[idx])
  with torch.no_grad():flat[idx]=old+eps
  lp=((x@w+b-target)**2).mean()
  with torch.no_grad():flat[idx]=old-eps
  lm=((x@w+b-target)**2).mean()
  with torch.no_grad():flat[idx]=old
  fd=(lp-lm)/(2*eps);ref=(cw.reshape(-1)[idx] if kind=='w' else cb[idx]);rows.append({'kind':kind,'index':idx,'finite_difference':float(fd),'closed_form':float(ref),'autograd':float((gw.reshape(-1)[idx] if kind=='w' else gb[idx])),'metrics':scalar_metrics(fd,ref)})
 torch.set_default_dtype(torch.float32);mx=max(r['metrics']['reference_relative_error'] for r in rows);return {'status':'pass' if mx<THRESH and all(abs(r['autograd']-r['closed_form'])<1e-10 for r in rows) else 'fail','dtype':'float64','rows':rows,'max_reference_relative_error':mx,'closed_form_bias_included':True,'threshold':THRESH}
def load():
 d=json.loads((QROOT/'dataset-manifest.json').read_text());r=next(x for x in d['rows'] if x['partition']=='train');enc=MiniLMEncoder(threads=1);x=torch.from_numpy(enc.encode(r['evidence_a']).astype(np.float32));eid=enc.identity();enc.close();tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();m.config.use_cache=False;ids=torch.tensor([[int(tok.eos_token_id)]+tok('Question: '+r['query']+' Answer:',add_special_tokens=True)['input_ids']]);target=torch.tensor([tok(' '+r['answer_a'],add_special_tokens=False)['input_ids'][0]]);base=m.get_input_embeddings()(ids).detach().float();eos=base[:,0,:].clone();b=init_bridge(bias=eos[0]);return r,x,eid,tok,m,b,target,base
def stage_b(x,b):
 y=b(x.unsqueeze(0));g=torch.arange(y.numel(),dtype=torch.float32).reshape_as(y)/y.numel();gw,gb=torch.autograd.grad(y,(b.weight,b.bias),g);ew=torch.outer(g[0],x);return {'status':'pass' if float((gw-ew).abs().max())<ABS_TOL and float((gb-g[0]).abs().max())<ABS_TOL else 'fail','weight':metrics(gw,ew),'bias':metrics(gb,g[0]),'output_shape':list(y.shape),'threshold':ABS_TOL}
def compare(m,hidden,target,depth):
 c=conventional(m,hidden,depth,target);lw=layerwise(m,hidden,depth,target);fo,fl=forward_only(m,hidden,depth,target);return {'loss':scalar_metrics(c['loss'],lw['loss']),'forward_only':scalar_metrics(c['loss'],float(fo)),'logits':metrics(c['logits'],lw['logits']),'gradient':metrics(c['grad'][:,0,:],lw['grad'][:,0,:]),'operation_counts':{'conventional_forward_layers':depth,'manual_forward_layers':depth,'manual_backward_vjps':depth,'forward_only_backward_vjps':0}}
def stage_c(m,b,x,base,target):
 slots={'eos':base[:,0,:].clone(),'initialized_bridge':b(x.unsqueeze(0)).detach()};results={}
 for name,slot in slots.items():
  hidden=torch.cat([slot.unsqueeze(1),base[:,1:,:]],dim=1).contiguous();results[name]={str(d):compare(m,hidden,target,d) for d in [0,1,2]}
 ok=all(v[str(d)]['gradient']['reference_relative_l2']<THRESH and v[str(d)]['gradient']['cosine']>COS_TOL for v in results.values() for d in [1,2]);return {'status':'pass' if ok else 'fail','inputs':results,'thresholds':{'relative_l2':THRESH,'cosine':COS_TOL}}
def stage_d(m,b,x,base,target):
 slot=b(x.unsqueeze(0)).detach();hidden=torch.cat([slot.unsqueeze(1),base[:,1:,:]],dim=1).contiguous();manual=layerwise(m,hidden,30,target)['grad'][:,0,:];aw=torch.outer(manual[0],x).reshape(-1);ab=manual[0];params=[b.weight.reshape(-1),b.bias.reshape(-1)];indices=[0,384,1000,10000,200000];rows=[];dirs=[]
 for eps in [1e-3,1e-2,1e-1]:
  for idx in indices:
   p=params[0] if idx<params[0].numel() else params[1];j=idx if idx<params[0].numel() else idx-params[0].numel();old=float(p[j])
   with torch.no_grad():p[j]=old+eps
   hp=torch.cat([b(x.unsqueeze(0)).detach().unsqueeze(1),base[:,1:,:]],dim=1);lp,_=forward_only(m,hp,30,target)
   with torch.no_grad():p[j]=old-eps
   hm=torch.cat([b(x.unsqueeze(0)).detach().unsqueeze(1),base[:,1:,:]],dim=1);lm,_=forward_only(m,hm,30,target)
   with torch.no_grad():p[j]=old
   num=(lp-lm)/(2*eps);ana=(aw[idx] if idx<aw.numel() else ab[idx-aw.numel()]);mm=scalar_metrics(num,ana);rows.append({'epsilon':eps,'parameter_index':idx,'numeric':float(num),'analytic':float(ana),'metrics':mm})
 mx=max(r['metrics']['reference_relative_error'] for r in rows);qualified_rows=[r for r in rows if r['epsilon']==.1];ok=all(r['metrics']['reference_relative_error']<THRESH or (abs(r['analytic'])<ABS_TOL and r['metrics']['absolute_error']<ABS_TOL) for r in qualified_rows);return {'status':'pass' if ok else 'fail','rows':rows,'qualified_epsilon':.1,'max_reference_relative_error_at_qualified_epsilon':max(r['metrics']['reference_relative_error'] for r in qualified_rows),'max_reference_relative_error_all_epsilons':mx,'manual_slot_gradient_norm':float(manual.norm()),'threshold':THRESH,'near_zero_absolute_tolerance':ABS_TOL,'note':'smaller epsilons are retained; BF16 forward quantization makes them diagnostic rather than qualification epsilons'}
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();a=stage_a();r,x,eid,t,m,b,target,base=load();bb=stage_b(x,b);cc=stage_c(m,b,x,base,target);dd=stage_d(m,b,x,base,target);stages={'stage_A':a,'stage_B':bb,'stage_C':cc,'stage_D':dd};qualified=all(v['status']=='pass' for v in stages.values());out={'schema':'LITE-008S-gradient-qualification-v1','status':'qualified' if qualified else 'failed','stages':stages,'thresholds':{'reference_relative_error':THRESH,'absolute_tolerance':ABS_TOL,'cosine':COS_TOL},'decoder_layers':30,'bridge':'Linear(384,576)','objective':'answer-token cross-entropy','optimizer_updates':0,'encoder_identity':eid,'decoder_artifact_identity':{'model_dir':str(MODEL)},'tokenizer_identity':{'tokenizer_dir':str(MODEL),'eos_token_id':int(t.eos_token_id)},'qualification_implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'acceptance_contract_identity':'LITE-008S-contract-v1','input_identity':{'feature_sha256':sha(x),'case_id':r['case_id']},'numerical_configuration':{'seed':SEED,'threads':1,'forward_only_finite_difference':True,'detached_position_state':True},'protected_final_holdout_accessed':False,'wall_seconds':time.monotonic()-st};(ROOT/'gradient-qualification-s.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'A':a['status'],'B':bb['status'],'C':cc['status'],'D':dd['status']}))
if __name__=='__main__':main()
