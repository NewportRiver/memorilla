"""LITE-008T bounded shared residual bridge training; exact T gate required."""
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import ids_for,tokid,objective,run_forward
ROOT=Path(__file__).resolve().parent;QROOT=ROOT.parent/'lite008q';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';Q=ROOT/'contrastive-qualification.json';LR=.0001;UPDATES=64;SEED=20261126
def gate():
 q=json.loads(Q.read_text());required=['decoder_hashes','tokenizer_hashes','encoder_identity','bridge_architecture','parameterization','objective_implementation_hash','decoder_differentiation_hash','numerical_configuration','acceptance_contract_identity']
 if q.get('status')!='qualified' or q.get('optimizer_updates')!=0 or any(k not in q for k in required):raise RuntimeError('FAIL_CLOSED: T qualification identity incomplete')
 if q.get('objective')!='answer_NLL + 0.5*softplus(-correct_margin)' or q.get('parameterization')!='residual_eos_plus_linear':raise RuntimeError('FAIL_CLOSED: training configuration mismatch')
 return q
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();q=gate();d=json.loads((QROOT/'dataset-manifest.json').read_text());e=MiniLMEncoder(threads=1);f={r['case_id']:e.encode(r['evidence_a']).astype(np.float32) for r in d['rows']};eid=e.identity();e.close();t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();eos=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0];W=torch.nn.Parameter(torch.zeros(576,384));opt=torch.optim.Adam([W],lr=LR);train=[r for r in d['rows'] if r['partition']=='train'];records=[];count=0
 for r in train:
  x=torch.from_numpy(f[r['case_id']]);a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);ids=ids_for(t,r['query'],1)
  for target in [True,False]:
   for _ in range(4):
    slot=(eos+W@x).unsqueeze(0);pre=objective(m,ids,slot,a,b,target,'contrastive');gslot=pre['grad'][0];W.grad=torch.outer(gslot,x);before=W.detach().clone();gn=float(W.grad.norm());torch.nn.utils.clip_grad_norm_([W],1.0);clipped=float(W.grad.norm());opt.step();opt.zero_grad(set_to_none=True);after=W.detach().clone();count+=1;post=run_forward(m,ids,(eos+W@x).unsqueeze(0),a,b);records.append({'update':count,'case_id':r['case_id'],'target':'A' if target else 'B','objective_pre':pre['loss'],'nll_pre':pre['nll'],'margin_pre':pre['margin'],'nll_a_post':post['nll_a'],'nll_b_post':post['nll_b'],'margin_post':post['margin'],'gradient_norm_before_clip':gn,'gradient_norm_after_clip':clipped,'parameter_displacement':float((after-before).norm()),'W_norm':float(W.norm()),'latent_norm':float((eos+W@x).norm()),'latent_displacement':float((W@x).norm()),'adam_m_norm':float(opt.state[W]['exp_avg'].norm()),'adam_v_norm':float(opt.state[W]['exp_avg_sq'].norm()),'rss_bytes':int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])*1024,'available_bytes':int(next(z for z in Path('/proc/meminfo').read_text().splitlines() if z.startswith('MemAvailable:')).split()[1])*1024})
 ck=ROOT/'shared-residual-checkpoint.npz';np.savez(ck,W=W.detach().numpy(),eos=eos.numpy());meta={'schema':'LITE-008T-shared-residual-checkpoint-v1','lr':LR,'updates':count,'parameterization':'z=EOS+W*x','checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'encoder_identity':eid};(ROOT/'shared-residual-checkpoint.json').write_text(json.dumps(meta,indent=2)+'\n');re=np.load(ck);reload_equal=bool(np.array_equal(re['W'],W.detach().numpy()) and np.array_equal(re['eos'],eos.numpy()));out={'schema':'LITE-008T-bridge-training-v1','status':'completed','qualification_artifact':str(Q),'parameterization':'residual_eos_plus_linear','lr':LR,'updates':count,'cumulative_authorized_updates_including_calibration':96,'trajectory':records,'checkpoint':str(ck),'checkpoint_sha256':meta['checkpoint_sha256'],'checkpoint_reload_equal':reload_equal,'encoder_identity':eid,'decoder_frozen':True,'protected_final_holdout_accessed':False,'wall_seconds':time.monotonic()-st};(ROOT/'bridge-training.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'updates':count,'reload':reload_equal,'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
