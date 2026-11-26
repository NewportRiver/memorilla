from __future__ import annotations
import hashlib,json,time,os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import run_forward,tokid,ids_for,MODEL
ROOT=Path(__file__).resolve().parents[2]; AUD=ROOT/'docs/memorilla-lite/audits'; HOME=Path.home(); R1=HOME/'.cache/memorilla-lite/LITE-008V-R1/original-generation.json'; F=HOME/'.cache/memorilla-lite/LITE-008V-R2/features.json'; B=HOME/'.cache/memorilla-lite/LITE-008V-R2/reconstruction.pt'; MAN=ROOT/'experiments/lite008p/dataset-manifest.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x): (AUD/n).write_text(json.dumps(x,indent=2)+'\n')
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1); started=time.monotonic(); td=json.loads(R1.read_text()); ff=json.loads(F.read_text()); man=json.loads(MAN.read_text()); rows={(r['case_id'], 'A'):r for r in man['rows']}; rows.update({(r['case_id'],'B'):r for r in man['rows']}); teachers={(r['case_id'],r['target']):torch.tensor(r['final_slots'][0],dtype=torch.float64) for r in td['results']}; feats={(r['case_id'],r['target']):torch.tensor(r['feature'],dtype=torch.float64) for r in ff['features']}; keys=list(teachers); assert set(keys)==set(feats)
 expected='73812ebd5810bee8115275f1dc6e291ace59895c41ff74951ebddb0721ac84fd'; assert sha(B)==expected; bundle=torch.load(B,map_location='cpu'); eos=bundle['eos'].double(); X=torch.stack([feats[k] for k in keys]); Y=torch.stack([teachers[k]-eos for k in keys]); tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True); model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True); model.requires_grad_(False); model.eval(); schedule=[0.,1e-6,1e-3,1e-2,.1,1.]
 folds=[]
 for entity in sorted(set(k[0] for k in keys)):
  hold=[i for i,k in enumerate(keys) if k[0]==entity]; train=[i for i in range(8) if i not in hold]; Xt=X[train];Yt=Y[train]; fold={'entity':entity,'held_out_keys':[keys[i] for i in hold],'training_keys':[keys[i] for i in train],'regularization':[]}
  for lam in schedule:
   A=torch.linalg.lstsq(Xt,Yt).solution if lam==0 else torch.linalg.solve(Xt.T@Xt+lam*torch.eye(X.shape[1],dtype=torch.float64),Xt.T@Yt); pred=X@A; rec={'lambda':lam,'weight_norm':float(A.norm()),'rows':[]}
   for i in hold:
    r=next(x for x in man['rows'] if x['case_id']==keys[i][0]); a=tokid(tok,r['answer_a']); b=tokid(tok,r['answer_b']); ids=ids_for(tok,r['query'],1); z=eos+pred[i]; o=run_forward(model,ids,z.float().reshape(1,-1),a,b); err=pred[i]-Y[i]; rec['rows'].append({'key':keys[i],'residual_mse':float((err*err).mean()),'residual_max_abs':float(err.abs().max()),'full_embedding_norm':float(z.norm()),'correct_answer_nll':o['nll_a'] if keys[i][1]=='A' else o['nll_b'],'margin':o['margin'],'preference':'A' if o['margin']>0 else 'B','reversal':('A' if o['margin']>0 else 'B')==keys[i][1]})
   fold['regularization'].append(rec)
  # controls once per held-out direction
  controls=[]
  for i in hold:
   r=next(x for x in man['rows'] if x['case_id']==keys[i][0]); a=tokid(tok,r['answer_a']);b=tokid(tok,r['answer_b']);ids=ids_for(tok,r['query'],1); mean=teachers and torch.stack(list(teachers.values())).mean(0).float(); wrong=feats[next(k for k in keys if k[0]!=keys[i][0])]; controls.append({'key':keys[i],'query_only':run_forward(model,ids,eos.float().reshape(1,-1),a,b),'mean_teacher':run_forward(model,ids,mean.reshape(1,-1),a,b),'mismatched_evidence':run_forward(model,ids,(eos.float()+wrong.float()@bundle['W'].float().T).reshape(1,-1),a,b)})
  fold['controls']=controls;folds.append(fold)
 save('LITE-008W-R2-HISTORICAL-FUNCTIONAL-CROSS-VALIDATION.json',{'status':'COMPLETED_HISTORICAL_DIAGNOSTIC','not_independent_acceptance':True,'schedule':schedule,'folds':folds,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-started})
 save('LITE-008W-R2-REGULARIZATION-ANALYSIS.json',{'status':'COMPLETED_HISTORICAL_DIAGNOSTIC','schedule':schedule,'selection_source':'historical folds only; no independent development selection','folds':folds,'optimizer_updates':0})
 print(json.dumps({'status':'completed','folds':len(folds),'seconds':time.monotonic()-started}))
if __name__=='__main__':main()
