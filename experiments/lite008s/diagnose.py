"""LITE-008S deterministic decoder VJP isolation; zero optimizer updates."""
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008n.layerwise_reference import masks,layer_call,fp_components,layerwise,conventional
from experiments.lite008q.bridge_train import init_bridge
ROOT=Path(__file__).resolve().parent;QROOT=ROOT.parent/'lite008q';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';SEED=20261126;EPS=[1e-3,1e-2,1e-1]
def h(t): return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def metrics(a,b):
 d=(a-b).float();aa=a.float();bb=b.float();an=float(aa.norm());bn=float(bb.norm());return {'absolute_max':float(d.abs().max()),'absolute_l2':float(d.norm()),'reference_relative_l2':float(d.norm()/max(an,1e-12)),'symmetric_relative_l2':float(d.norm()/max((an+bn)/2,1e-12)),'cosine':float(torch.nn.functional.cosine_similarity(aa.flatten(),bb.flatten(),dim=0)),'norm_ratio':float(bn/max(an,1e-12))}
def forward_only(m,hidden,n,target):
 with torch.no_grad():
  posids,pos,mask,pe=masks(m,hidden);x=hidden.float()
  for i in range(n):x=layer_call(copy.deepcopy(m.model.layers[i]).float(),x,mask,posids,pos,pe)
  norm,ow=fp_components(m);logits=torch.nn.functional.linear(norm(x)[:,-1,:],ow).float();return torch.nn.functional.cross_entropy(logits,target),logits

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();d=json.loads((QROOT/'dataset-manifest.json').read_text());r=next(x for x in d['rows'] if x['partition']=='train');enc=MiniLMEncoder(threads=1);feat=torch.from_numpy(enc.encode(r['evidence_a']).astype(np.float32));eid=enc.identity();enc.close();tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();m.config.use_cache=False
 ids=torch.tensor([[int(tok.eos_token_id)]+tok('Question: '+r['query']+' Answer:',add_special_tokens=True)['input_ids']]);target=torch.tensor([tok(' '+r['answer_a'],add_special_tokens=False)['input_ids'][0]]);base=m.get_input_embeddings()(ids).detach().float();eos=base[:,0,:].clone();torch.manual_seed(SEED);random=torch.randn_like(eos);random*=eos.norm()/random.norm();bridge=init_bridge(bias=eos[0]);bridge_out=bridge(feat.unsqueeze(0)).detach();q=np.load(Path.home()/'.cache/memorilla-lite/LITE-008Q/stable-contrastive-bridge.npz');qbridge=torch.nn.Linear(384,576);qbridge.load_state_dict({'weight':torch.from_numpy(q['weight']),'bias':torch.from_numpy(q['bias'])});high=qbridge(feat.unsqueeze(0)).detach();opt,_=__import__('experiments.lite008t.qualification',fromlist=['extract_p']).extract_p();cases={'eos':eos,'eos_perturbed':eos+torch.linspace(-1,1,576).unsqueeze(0)*1e-3,'random_scale_matched':random,'initialized_bridge':bridge_out,'optimized_P':opt,'historical_Q_bridge':high}
 out={'schema':'LITE-008S-minimal-reproduction-v1','status':'completed','optimizer_updates':0,'seed':SEED,'decoder_layers':30,'query_ids':ids.tolist(),'target':target.tolist(),'encoder_identity':eid,'cases':{},'depths':[0,1,2],'epsilons':EPS,'wall_seconds':None}
 for name,slot in cases.items():
  hidden=torch.cat([slot.unsqueeze(1),base[:,1:,:]],dim=1).contiguous();entry={'input':{'sha256':h(slot),'dtype':str(slot.dtype),'shape':list(slot.shape),'strides':list(slot.stride()),'contiguous':slot.is_contiguous(),'norm':float(slot.norm()),'storage_offset':slot.storage_offset(),'storage_aliases_hidden':slot.storage().data_ptr()==hidden.storage().data_ptr()}}
  for depth in [0,1,2]:
   c=conventional(m,hidden,depth,target);lw=layerwise(m,hidden,depth,target);fo,fl=forward_only(m,hidden,depth,target);reps=[float(forward_only(m,hidden,depth,target)[0]) for _ in range(3)];v=torch.linspace(-1,1,576).reshape(1,576);v/=v.norm();num=[]
   for ep in EPS:
    hp=hidden.detach().clone();hm=hidden.detach().clone();hp[:,0,:]+=ep*v;hm[:,0,:]-=ep*v;lp,_=forward_only(m,hp,depth,target);lm,_=forward_only(m,hm,depth,target);num.append(float((lp-lm)/(2*ep)))
   entry[str(depth)]={'loss_conventional':c['loss'],'loss_manual':lw['loss'],'loss_forward_only':float(fo),'forward_conventional_vs_forward_only':metrics(c['logits'],fl),'repeated_forward_losses':reps,'repeated_forward_max_abs_delta':max(reps)-min(reps),'conventional_vs_manual_gradient':metrics(c['grad'][:,0,:],lw['grad'][:,0,:]),'directional_derivatives':{'analytic_conventional':[float((c['grad'][:,0,:]*v).sum())],'analytic_manual':[float((lw['grad'][:,0,:]*v).sum())],'finite_difference':num}}
  out['cases'][name]=entry
 out['wall_seconds']=time.monotonic()-st;(ROOT/'minimal_reproduction.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'cases':len(cases),'status':out['status'],'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
