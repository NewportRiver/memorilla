"""LITE-008T contrastive gradient-chain qualification; no optimizer updates."""
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008n.layerwise_reference import masks,layer_call,fp_components,layerwise
from experiments.lite008q.bridge_train import init_bridge
ROOT=Path(__file__).resolve().parent;QROOT=ROOT.parent/'lite008q';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';AUD=ROOT.parent.parent/'docs/memorilla-lite/audits';SEED=20261126;THRESH=.10;ABS_TOL=1e-3;EPS=[.01,.1]
def shab(b):return hashlib.sha256(b).hexdigest()
def sha(t):return shab(t.detach().contiguous().cpu().numpy().tobytes())
def metric(a,b):
 a=a.detach().float();b=b.detach().float();d=a-b;an=float(a.norm());bn=float(b.norm());return {'absolute_max':float(d.abs().max()),'absolute_l2':float(d.norm()),'reference_relative_l2':float(d.norm()/max(an,1e-12)),'symmetric_relative_l2':float(d.norm()/max((an+bn)/2,1e-12)),'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),'gradient_norm_ratio':float(bn/max(an,1e-12))}
def scalar(a,b):
 a=float(a);b=float(b);d=abs(a-b);return {'absolute_error':d,'reference_relative_error':d/max(abs(a),1e-12),'symmetric_relative_error':d/max((abs(a)+abs(b))/2,1e-12)}
def objective_value(logits,a,b,target):
 lp=torch.log_softmax(logits.float(),-1); margin=lp[0,a]-lp[0,b]; signed=margin if target else -margin; nll=-lp[0,a] if target else -lp[0,b];return nll+.5*torch.nn.functional.softplus(-signed)
def forward_only(m,hidden,target,a,b):
 with torch.no_grad():
  posids,pos,mask,pe=masks(m,hidden.detach().float());x=hidden.detach().float()
  for i in range(30):x=layer_call(copy.deepcopy(m.model.layers[i]).float(),x,mask,posids,pos,pe)
  norm,ow=fp_components(m);logits=torch.nn.functional.linear(norm(x)[:,-1,:],ow).float();return float(objective_value(logits,a,b,target)),logits
def conventional(m,hidden,target,a,b):
 x0=hidden.detach().float().requires_grad_(True);x=x0;posids,pos,mask,pe=masks(m,x0.detach());
 for i in range(30):x=layer_call(copy.deepcopy(m.model.layers[i]).float(),x,mask,posids,pos,pe)
 norm,ow=fp_components(m);logits=torch.nn.functional.linear(norm(x)[:,-1,:],ow).float();loss=objective_value(logits,a,b,target);g=torch.autograd.grad(loss,x0)[0].detach();return float(loss),g

def extract_p():
 p=AUD/'LITE-008P-CONVERGENCE-RESULTS.json'
 if not p.is_file():raise RuntimeError('MISSING_REQUIRED_ARTIFACT: LITE-008P convergence report')
 d=json.loads(p.read_text());found=[]
 def walk(x):
  if isinstance(x,dict):
   if x.get('case_id')=='case_00' and x.get('target')=='A' and x.get('updates')==16 and x.get('lr')==.05 and 'final_slots' in x:found.append(x['final_slots'][0])
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(d)
 if not found:raise RuntimeError('MISSING_REQUIRED_ARTIFACT: recoverable LITE-008P final slot not found')
 t=torch.tensor(found[0],dtype=torch.float32).reshape(1,576);return t,{'source':str(p),'case_id':'case_00','target':'A','updates':16,'lr':.05,'sha256':sha(t),'norm':float(t.norm())}
def load():
 man=json.loads((QROOT/'dataset-manifest.json').read_text());r=next(x for x in man['rows'] if x['partition']=='train');enc=MiniLMEncoder(threads=1);feat=torch.from_numpy(enc.encode(r['evidence_a']).astype(np.float32));eid=enc.identity();enc.close();tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();m.config.use_cache=False;ids=torch.tensor([[int(tok.eos_token_id)]+tok('Question: '+r['query']+' Answer:',add_special_tokens=True)['input_ids']]);a=tok(' '+r['answer_a'],add_special_tokens=False)['input_ids'][0];b=tok(' '+r['answer_b'],add_special_tokens=False)['input_ids'][0];base=m.get_input_embeddings()(ids).detach().float();eos=base[:,0,:].clone();bridge=init_bridge(bias=eos[0]);p,pid=extract_p();return r,feat,eid,tok,m,bridge,p,pid,ids,a,b,base
def qualify_case(m,bridge,slot,base,ids,a,b,target,label):
 hidden=torch.cat([slot.unsqueeze(1),base[:,1:,:]],1).contiguous();cl,cg=conventional(m,hidden,target,a,b);ml,mg=layerwise(m,hidden,30,torch.tensor([a if target else b]))['loss'],layerwise(m,hidden,30,torch.tensor([a if target else b]))['grad']
 # layerwise objective is NLL-only, so recompute exact contrastive manual gradient via retained layer states
 posids,pos,mask,pe=masks(m,hidden);states=[hidden]
 x=hidden
 for i in range(30):x=layer_call(copy.deepcopy(m.model.layers[i]).float(),x,mask,posids,pos,pe).detach();states.append(x)
 norm,ow=fp_components(m);h=states[-1].detach().requires_grad_(True);logits=torch.nn.functional.linear(norm(h)[:,-1,:],ow).float();loss=objective_value(logits,a,b,target);gh=torch.autograd.grad(loss,h)[0].detach()
 for i in reversed(range(30)):
  inp=states[i].detach().requires_grad_(True);out=layer_call(copy.deepcopy(m.model.layers[i]).float(),inp,mask,posids,pos,pe);gh=torch.autograd.grad(out,inp,grad_outputs=gh)[0].detach()
 mg=gh;rows=[]
 for eps in EPS:
  hp=hidden.detach().clone();hm=hidden.detach().clone();direction=torch.linspace(-1,1,576).reshape(1,576);direction/=direction.norm();hp[:,0]+=eps*direction;hm[:,0]-=eps*direction;lp,_=forward_only(m,hp,target,a,b);lm,_=forward_only(m,hm,target,a,b);fd=(lp-lm)/(2*eps);ana=float((mg[:,0]*direction).sum());rows.append({'epsilon':eps,'numeric':fd,'analytic':ana,'metrics':scalar(fd,ana)})
 return {'label':label,'target':'A' if target else 'B','external_embedding':metric(cg[:,0],mg[:,0]),'conventional_manual_loss':scalar(cl,float(loss)),'conventional_forward_only':scalar(cl,forward_only(m,hidden,target,a,b)[0]),'external_directional':rows,'operation_counts':{'conventional_backward':1,'manual_backward_vjps':30,'forward_only_backward_vjps':0}}
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();r,x,eid,t,m,b,p,pid,ids,a,bb,base=load();init=b(x.unsqueeze(0)).detach();pert=init+torch.linspace(-1,1,576).reshape(1,576)*1e-3;cases=[('initialized_bridge',init),('perturbed_bridge',pert),('P_embedding',p)];results=[]
 for label,slot in cases:
  for target in [True,False]:results.append(qualify_case(m,b,slot,base,ids,a,bb,target,label))
 # bridge parameter directions on initialized bridge, exact objective finite differences
 hidden_slot=init;manual_slot=results[0];# use independently recomputed external gradient for both targets
 param_rows=[]
 for target in [True,False]:
  hidden=torch.cat([hidden_slot.unsqueeze(1),base[:,1:]],1);posids,pos,mask,pe=masks(m,hidden);states=[hidden];z=hidden
  for i in range(30):z=layer_call(copy.deepcopy(m.model.layers[i]).float(),z,mask,posids,pos,pe).detach();states.append(z)
  norm,ow=fp_components(m);h=states[-1].detach().requires_grad_(True);logits=torch.nn.functional.linear(norm(h)[:,-1,:],ow).float();loss=objective_value(logits,a,bb,target);gh=torch.autograd.grad(loss,h)[0].detach()
  for i in reversed(range(30)):
   inp=states[i].detach().requires_grad_(True);out=layer_call(copy.deepcopy(m.model.layers[i]).float(),inp,mask,posids,pos,pe);gh=torch.autograd.grad(out,inp,grad_outputs=gh)[0].detach()
  gw=torch.outer(gh[0,0],x);gb=gh[0,0];
  for idx in [0,1000,221759]:
   isw=idx<gw.numel();pflat=b.weight.reshape(-1) if isw else b.bias.reshape(-1);j=idx if isw else idx-gw.numel();old=float(pflat[j]);
   for eps in EPS:
    with torch.no_grad():pflat[j]=old+eps
    lp=forward_only(m,torch.cat([b(x.unsqueeze(0)).detach().unsqueeze(1),base[:,1:]],1),target,a,bb)[0]
    with torch.no_grad():pflat[j]=old-eps
    lm=forward_only(m,torch.cat([b(x.unsqueeze(0)).detach().unsqueeze(1),base[:,1:]],1),target,a,bb)[0]
    with torch.no_grad():pflat[j]=old
    ana=float((gw.reshape(-1)[idx] if isw else gb[j]));param_rows.append({'target':'A' if target else 'B','kind':'weight' if isw else 'bias','index':idx,'epsilon':eps,'numeric':(lp-lm)/(2*eps),'analytic':ana,'metrics':scalar((lp-lm)/(2*eps),ana)})
 out={'schema':'LITE-008T-contrastive-gradient-qualification-v1','status':'completed','optimizer_updates':0,'objective':'answer_NLL + 0.5*softplus(-correct_margin)','cases':results,'bridge_parameter_rows':param_rows,'thresholds':{'reference_relative_error':THRESH,'near_zero_absolute_tolerance':ABS_TOL},'encoder_identity':eid,'p_embedding_identity':pid,'input_identity':{'feature_sha256':sha(x),'query_ids':ids.tolist()},'decoder_hashes':{n:shab((MODEL/n).read_bytes()) for n in ['model.safetensors','tokenizer.json','config.json']},'tokenizer_hashes':{n:shab((MODEL/n).read_bytes()) for n in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt']},'bridge_architecture':'Linear(384,576)','parameterization':'residual_eos_plus_linear','objective_implementation_hash':shab((ROOT.parent/'lite008p/p_experiment.py').read_bytes()),'decoder_differentiation_hash':shab((ROOT.parent/'lite008n/layerwise_reference.py').read_bytes()),'acceptance_contract_identity':'LITE-008T-contract-v1','numerical_configuration':{'seed':SEED,'threads':1,'dtype':'bf16-stored-fp32-master','sequence_length_max':32,'external_slots':1,'finite_difference_epsilons':EPS},'implementation_hash':shab(Path(__file__).read_bytes()),'wall_seconds':time.monotonic()-st};ok=all(r['external_embedding']['reference_relative_l2']<THRESH and r['external_embedding']['cosine']>.999 for r in results);ok=ok and all(r['metrics']['reference_relative_error']<THRESH or (abs(r['analytic'])<ABS_TOL and r['metrics']['absolute_error']<ABS_TOL) for r in param_rows if r['epsilon']==.1);out['status']='qualified' if ok else 'failed';(ROOT/'contrastive-qualification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'updates':0,'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
