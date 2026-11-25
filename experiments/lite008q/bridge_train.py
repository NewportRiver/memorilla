"""LITE-008Q shared MiniLM(384)->decoder(576) bridge pilot."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import MODEL,ids_for,tokid,objective,run_forward
ROOT=Path(__file__).resolve().parent;CACHE=Path.home()/'.cache/memorilla-lite/LITE-008Q';TOKDIR=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
def append(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def require_qualification(path,expected_objective=None):
 if not path: raise RuntimeError('FAIL_CLOSED: --qualification is required before bridge training')
 p=Path(path)
 if not p.is_file(): raise RuntimeError(f'FAIL_CLOSED: qualification artifact missing: {p}')
 q=json.loads(p.read_text())
 if q.get('schema')!='LITE-008S-gradient-qualification-v1' or q.get('status')!='qualified': raise RuntimeError('FAIL_CLOSED: gradient qualification is not qualified')
 for stage in ('stage_A','stage_B','stage_C','stage_D'):
  if q.get('stages',{}).get(stage,{}).get('status')!='pass': raise RuntimeError(f'FAIL_CLOSED: {stage} did not pass')
 if q.get('optimizer_updates',0)!=0: raise RuntimeError('FAIL_CLOSED: qualification artifact records optimizer updates')
 required=('decoder_artifact_identity','tokenizer_identity','encoder_identity','bridge','qualification_implementation_sha256','objective','numerical_configuration','input_identity')
 if any(not q.get(k) for k in required): raise RuntimeError('FAIL_CLOSED: qualification identity fields incomplete')
 if expected_objective is not None and q.get('objective')!=expected_objective: raise RuntimeError('FAIL_CLOSED: objective/configuration mismatch')
 return q
def load_data():
 d=json.loads((ROOT/'dataset-manifest.json').read_text());enc=MiniLMEncoder(threads=1);feats={}
 for r in d['rows']:
  feats[r['case_id']]=enc.encode(r['evidence_a']).astype(np.float32);feats[r['case_id']+'B']=enc.encode(r['evidence_b']).astype(np.float32)
 ident=enc.identity();enc.close();return d,feats,ident
def init_bridge(seed=20261126,bias=None):
 torch.manual_seed(seed);b=torch.nn.Linear(384,576)
 with torch.no_grad():
  b.weight.normal_(0,.0005)
  if bias is None:b.bias.zero_()
  else:b.bias.copy_(bias)
 return b
def example_rows(d,partition):
 out=[]
 for r in d['rows']:
  if r['partition']==partition:out += [(r,True),(r,False)]
 return out
def train_bridge(m,t,d,feats,mode,updates,lr,ledger,qualification):
 require_qualification(qualification,mode)
 eos=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0];b=init_bridge(bias=eos);initial={k:v.detach().clone() for k,v in b.state_dict().items()};opt=torch.optim.Adam(b.parameters(),lr=lr);examples=example_rows(d,'train');traj=[];completed=0
 for u in range(updates):
  r,target=examples[u%len(examples)];x=torch.from_numpy(feats[r['case_id'] if target else r['case_id']+'B']).unsqueeze(0);a=tokid(t,r['answer_a']);bb=tokid(t,r['answer_b']);ids=ids_for(t,r['query'],1);slot=b(x);pre=objective(m,ids,slot.detach(),a,bb,target,mode);b.zero_grad(set_to_none=True);slot.backward(pre['grad']);gn=float(torch.cat([p.grad.reshape(-1) for p in b.parameters() if p.grad is not None]).norm());torch.nn.utils.clip_grad_norm_(b.parameters(),1.0);opt.step();completed+=1;post=run_forward(m,ids,b(x).detach(),a,bb);rec={'experiment':'LITE-008Q','mode':mode,'update':completed,'case_id':r['case_id'],'direction':'A' if target else 'B','objective_pre':pre['loss'],'nll_pre':pre['nll'],'margin_pre':pre['margin'],'nll_a_post':post['nll_a'],'nll_b_post':post['nll_b'],'margin_post':post['margin'],'bridge_gradient_norm':gn,'bridge_output_norm':float(b(x).detach().norm()),'decoder_forwards':pre['forwards'],'decoder_layer_recomputations':pre['recomputations'],'decoder_backward_vjps':pre['vjps'],'rss_bytes':int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])*1024,'available_bytes':int(next(z for z in Path('/proc/meminfo').read_text().splitlines() if z.startswith('MemAvailable:')).split()[1])*1024};append(ledger,rec);traj.append(rec)
 return b,initial,traj
def train_soft(m,t,d,feats,updates,lr):
 p=torch.nn.Parameter(torch.zeros(576));opt=torch.optim.Adam([p],lr=lr);ex=example_rows(d,'train');hist=[]
 with torch.no_grad():p.copy_(m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0])
 for u in range(updates):
  r,target=ex[u%len(ex)];ids=ids_for(t,r['query'],1);a=tokid(t,r['answer_a']);bb=tokid(t,r['answer_b']);pre=objective(m,ids,p.detach().unsqueeze(0),a,bb,target,'contrastive');p.grad=pre['grad'][0];torch.nn.utils.clip_grad_norm_([p],1.0);opt.step();opt.zero_grad(set_to_none=True);hist.append({'update':u+1,'case_id':r['case_id'],'margin':run_forward(m,ids,p.detach().unsqueeze(0),a,bb)['margin']})
 return p.detach(),hist
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['contrastive','nll','soft'],required=True);ap.add_argument('--updates',type=int,default=48);ap.add_argument('--tag',default='q');ap.add_argument('--lr',type=float,default=.05);ap.add_argument('--qualification');args=ap.parse_args();qualification=require_qualification(args.qualification);torch.set_num_threads(1);st=time.monotonic();CACHE.mkdir(parents=True,exist_ok=True);d,feats,encid=load_data();t=AutoTokenizer.from_pretrained(str(TOKDIR),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();ledger=CACHE/'update-ledger.jsonl'
 if args.mode=='soft':require_qualification(args.qualification,args.mode);p,hist=train_soft(m,t,d,feats,args.updates,args.lr);np.save(CACHE/f'{args.tag}-soft.npy',p.numpy());out={'schema':'LITE-008Q-soft-prompt-training-v1','mode':'soft','updates':args.updates,'history':hist,'encoder_identity':encid}
 else:
  b,initial,tr=train_bridge(m,t,d,feats,args.mode,args.updates,args.lr,ledger,args.qualification);arr={'weight':b.weight.detach().numpy(),'bias':b.bias.detach().numpy()};np.savez(CACHE/f'{args.tag}-bridge.npz',**arr);h=hashlib.sha256();h.update(arr['weight'].tobytes());h.update(arr['bias'].tobytes());out={'schema':'LITE-008Q-bridge-training-v1','mode':args.mode,'updates':args.updates,'learning_rate':args.lr,'qualification_artifact':args.qualification,'parameter_count':221760,'checkpoint':str(CACHE/f'{args.tag}-bridge.npz'),'checkpoint_sha256':h.hexdigest(),'encoder_identity':encid,'trajectory':tr,'initial_weight_sha256':hashlib.sha256(initial['weight'].numpy().tobytes()).hexdigest(),'decoder_parameters_frozen':True}
 out['wall_seconds']=time.monotonic()-st;(ROOT/f'{args.tag}-training.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'tag':args.tag,'updates':args.updates,'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
