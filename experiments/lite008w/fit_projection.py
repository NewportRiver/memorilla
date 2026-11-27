import json,hashlib,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import run_forward,tokid,ids_for,MODEL
ROOT=Path(__file__).resolve().parents[2];AUD=ROOT/'docs/memorilla-lite/audits';HOME=Path.home(); CACHE=HOME/'.cache/memorilla-lite/LITE-008W-R3'; TD=CACHE/'teachers'; F=HOME/'.cache/memorilla-lite/LITE-008W-R2/features.json'; MAN=AUD/'LITE-008W-R2-DATASET-MANIFEST.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic(); d=json.loads(MAN.read_text()); fd=json.loads(F.read_text()); eligible=['wtrain_00','wtrain_01','wtrain_02','wtrain_04','wtrain_05']; rows={r['id']:r for r in d['rows']}; feats={(x['id'],x['target']):torch.tensor(x['feature'],dtype=torch.float64) for x in fd['features']}; arts=[]
 for p in sorted(TD.glob('*.teacher.pt')):
  x=torch.load(p,map_location='cpu'); key=(x['payload']['case_id'],x['payload']['target']);
  if key[0] in eligible: arts.append((key,x))
 if len(arts)!=10: raise RuntimeError(f'expected 10 teachers, got {len(arts)}')
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval(); eos=m.get_input_embeddings()(torch.tensor([[tok.eos_token_id]])).detach().float()[0,0].double(); X=torch.stack([feats[k] for k,x in arts]); Y=torch.stack([x['embedding'].double().reshape(-1)-eos for k,x in arts]); schedule=[0.,1e-6,1e-3,1e-2,.1,1.]; configs={};
 for lam in schedule:
  A=torch.linalg.lstsq(X,Y).solution if lam==0 else torch.linalg.solve(X.T@X+lam*torch.eye(X.shape[1],dtype=torch.float64),X.T@Y); W=A.T; pred=X@A; configs[str(lam)]={'lambda':lam,'weight_norm':float(W.norm()),'residual_mse':float(((pred-Y)**2).mean()),'residual_max_abs':float((pred-Y).abs().max()),'full_embedding_norm_mean':float((pred+eos).norm(dim=1).mean()),'W':W.float()}
 # fixed selection from historical diagnostic: lowest lambda with stable finite training fit; no independent data used.
 selected=1e-3 if str(1e-3) in configs else 0.; sel=configs[str(selected)]; atomic=CACHE/'projection_selected.pt'; torch.save({'W':sel['W'],'eos':eos.float(),'lambda':selected,'manifest_sha256':sha(MAN),'feature_sha256':sha(F),'teacher_keys':[k for k,x in arts]},atomic)
 reload=torch.load(atomic,map_location='cpu'); train_rows=[]
 for i,(k,x) in enumerate(arts):
  r=rows[k[0]]; a=tokid(tok,r['answer_a']);b=tokid(tok,r['answer_b']); z=eos.float()+reload['W']@feats[k].float(); o=run_forward(m,ids_for(tok,r['query'],1),z.reshape(1,-1),a,b); train_rows.append({'key':k,'margin':o['margin'],'nll_a':o['nll_a'],'nll_b':o['nll_b'],'preference':'A' if o['margin']>0 else 'B','reversal':('A' if o['margin']>0 else 'B')==k[1],'latent_norm':float(z.norm())})
 report={'status':'COMPLETED','selected_lambda':selected,'selection_rule':'historical diagnostic plus training/calibration evidence only; independent data excluded','configs':{k:{kk:vv for kk,vv in v.items() if kk!='W'} for k,v in configs.items()},'selected_artifact':str(atomic),'selected_artifact_sha256':sha(atomic),'training_rows':train_rows,'training_reversals':sum(x['reversal'] for x in train_rows),'optimizer_updates':0,'algebraic_solver_operations':6,'wall_seconds':time.monotonic()-st}; (AUD/'LITE-008W-R3-MATHEMATICAL-LEARNING-RESULTS.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'selected_lambda':selected,'training_reversals':report['training_reversals'],'artifact':str(atomic),'seconds':report['wall_seconds']}))
if __name__=='__main__':main()
