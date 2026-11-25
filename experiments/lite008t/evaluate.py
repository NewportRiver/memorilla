"""LITE-008T fresh development evaluation and matched inference controls; no updates."""
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import ids_for,tokid,run_forward
ROOT=Path(__file__).resolve().parent;QROOT=ROOT.parent/'lite008q';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';CK=ROOT/'shared-residual-checkpoint.npz';SEED=20261126
def main():
 torch.set_num_threads(1);st=time.monotonic();d=json.loads((QROOT/'dataset-manifest.json').read_text());dev=[r for r in d['rows'] if r['partition']=='development'];train=[r for r in d['rows'] if r['partition']=='train'];e=MiniLMEncoder(threads=1);features={};
 for r in d['rows']:
  features[r['case_id']]=torch.from_numpy(e.encode(r['evidence_a']).astype(np.float32));features[r['case_id']+'B']=torch.from_numpy(e.encode(r['evidence_b']).astype(np.float32))
 eid=e.identity();e.close();t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();z=np.load(CK);W=torch.from_numpy(z['W']);eos=torch.from_numpy(z['eos']);mean=torch.stack([features[r['case_id']] for r in train]).mean(0);rows=[];rng=torch.Generator().manual_seed(SEED)
 for i,r in enumerate(dev):
  a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);ids=ids_for(t,r['query'],1);other=dev[(i+1)%len(dev)];rnd=torch.randn(eos.shape,generator=rng);rnd=rnd/rnd.norm()*eos.norm();slots={'learned_A':eos+W@features[r['case_id']],'learned_B':eos+W@features[r['case_id']+'B'], 'query_only':eos,'zero':torch.zeros_like(eos),'scale_matched_random':rnd,'untrained_bridge':eos,'mismatched_evidence':eos+W@features[other['case_id']],'mean_training_memory':eos+W@mean}
  for name,slot in slots.items():
   v=run_forward(m,ids,slot.unsqueeze(0),a,b);rows.append({'case_id':r['case_id'],'evidence_target':'A' if name.endswith('_A') or name in ['query_only','zero','scale_matched_random','untrained_bridge','mean_training_memory'] else 'B','control':name,'margin':v['margin'],'nll_a':v['nll_a'],'nll_b':v['nll_b'],'greedy_id':v['greedy_id'],'latent_norm':float(slot.norm())})
 # score learned with evidence A/B separately; same query, reversed factual assignment
 pairs=[]
 for r in dev:
  a=next(x for x in rows if x['case_id']==r['case_id'] and x['control']=='learned_A');b=next(x for x in rows if x['case_id']==r['case_id'] and x['control']=='learned_B');pairs.append({'case_id':r['case_id'],'margin_A':a['margin'],'margin_B':b['margin'],'reversal':a['margin']>0 and b['margin']<0})
 out={'schema':'LITE-008T-independent-development-v1','status':'completed','pairs':pairs,'rows':rows,'controls':['query_only','zero','scale_matched_random','untrained_bridge','mismatched_evidence','mean_training_memory'],'ordinary_textual_evidence':'historical LITE-008O textual prerequisite only; no new textual optimizer/control updates','memory_independent_soft_prompt':'not executed; reserved control would require optimizer updates','threshold':'at least 6/8 reversals and improvement over untrained/memory-independent controls','reversal_count':sum(x['reversal'] for x in pairs),'pair_count':len(pairs),'checkpoint_sha256':hashlib.sha256(CK.read_bytes()).hexdigest(),'encoder_identity':eid,'optimizer_updates':0,'protected_final_holdout_accessed':False,'wall_seconds':time.monotonic()-st};(ROOT/'independent-development.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'reversals':out['reversal_count'],'pairs':out['pair_count'],'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
