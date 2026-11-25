"""LITE-008T bounded residual-bridge calibration; authorized after contrastive qualification."""
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import ids_for,tokid,objective,run_forward
from experiments.lite008q.bridge_train import init_bridge
ROOT=Path(__file__).resolve().parent;QROOT=ROOT.parent/'lite008q';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';Q=ROOT/'contrastive-qualification.json';SEED=20261126;MAX_UPDATES=256
def load():
 d=json.loads((QROOT/'dataset-manifest.json').read_text());e=MiniLMEncoder(threads=1);f={r['case_id']:e.encode(r['evidence_a']).astype(np.float32) for r in d['rows']};eid=e.identity();e.close();t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();eos=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0];return d,f,eid,t,m,eos
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();q=json.loads(Q.read_text());assert q.get('status')=='qualified' and q.get('optimizer_updates')==0,'FAIL_CLOSED qualification';d,f,eid,t,m,eos=load();rows=[r for r in d['rows'] if r['partition']=='calibration'];arms=[];total=0
 for lr in [.001,.0001]:
  W=torch.nn.Parameter(torch.zeros(576,384));opt=torch.optim.Adam([W],lr=lr);arm={'lr':lr,'parameterization':'z=EOS+W*x','initial_W_norm':float(W.norm()),'updates':[]}
  for r in rows[:2]:
   for target in [True,False]:
    x=torch.from_numpy(f[r['case_id']]).unsqueeze(0);a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);ids=ids_for(t,r['query'],1)
    for u in range(4):
     slot=(eos+W@x[0]).unsqueeze(0);pre=objective(m,ids,slot,a,b,target,'contrastive');gslot=pre['grad'][0];g=torch.outer(gslot,x[0]);W.grad=g.detach().clone();before=W.detach().clone();gn=float(g.norm());torch.nn.utils.clip_grad_norm_([W],1.0);clipped=float(W.grad.norm());opt.step();opt.zero_grad(set_to_none=True);after=W.detach().clone();post=run_forward(m,ids,(eos+W@x[0]).unsqueeze(0),a,b);state=opt.state[W];total+=1;arm['updates'].append({'update':total,'case_id':r['case_id'],'target':'A' if target else 'B','lr':lr,'objective_pre':pre['loss'],'nll_pre':pre['nll'],'margin_pre':pre['margin'],'nll_a_post':post['nll_a'],'nll_b_post':post['nll_b'],'margin_post':post['margin'],'gradient_norm_before_clip':gn,'gradient_norm_after_clip':clipped,'parameter_displacement':float((after-before).norm()),'cumulative_W_displacement':float((after).norm()),'latent_norm':float((eos+W@x[0]).norm()),'latent_displacement_from_eos':float((W@x[0]).norm()),'adam_m_norm':float(state['exp_avg'].norm()),'adam_v_norm':float(state['exp_avg_sq'].norm()),'rss_bytes':int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])*1024,'available_bytes':int(next(z for z in Path('/proc/meminfo').read_text().splitlines() if z.startswith('MemAvailable:')).split()[1])*1024})
  arm['final_W_norm']=float(W.norm());arm['final_checkpoint_sha256']=hashlib.sha256(W.detach().numpy().tobytes()).hexdigest();arms.append(arm)
 out={'schema':'LITE-008T-optimization-stability-v1','status':'completed','qualification_artifact':str(Q),'arms':arms,'total_optimizer_updates':total,'max_authorized_updates':MAX_UPDATES,'selected_configuration':'not selected until calibration review','encoder_identity':eid,'decoder_parameters_frozen':True,'protected_final_holdout_accessed':False,'wall_seconds':time.monotonic()-st};(ROOT/'stability-calibration.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'updates':total,'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
