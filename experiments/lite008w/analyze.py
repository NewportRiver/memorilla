from pathlib import Path
import json, hashlib, time, os, sys
import torch
ROOT=Path(__file__).resolve().parents[2]
AUD=ROOT/'docs/memorilla-lite/audits'
CACHE=Path.home()/'.cache/memorilla-lite/LITE-008W'
CACHE.mkdir(parents=True,exist_ok=True)
R1=Path.home()/'.cache/memorilla-lite/LITE-008V-R1/original-generation.json'
F=Path.home()/'.cache/memorilla-lite/LITE-008V-R2/features.json'

def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def dump(name,x): (AUD/name).write_text(json.dumps(x,indent=2)+'\n')
def metrics(pred,y):
 e=pred-y
 return {'mse':float((e*e).mean()),'max_abs':float(e.abs().max()),'relative_l2':float(e.norm()/y.norm()),'pred_norm_mean':float(pred.norm(dim=1).mean())}

def main():
 torch.set_num_threads(1); torch.set_num_interop_threads(1)
 d=json.loads(R1.read_text()); f=json.loads(F.read_text())
 # Permanent content-bound identity contract: never infer association from row position.
 teacher_by_key={(r['case_id'],r['target']):r for r in d['results']}
 feature_by_key={(r['case_id'],r['target']):r for r in f['features']}
 keys=list(teacher_by_key)
 if set(keys)!=set(feature_by_key) or len(keys)!=8: raise RuntimeError('FAIL_CLOSED: teacher/feature identity key mismatch')
 if sha(F)!='4dbceffce6a1002640fa4347779844c7b3bd04f62edc29987f0b44cf60fb0b4c': raise RuntimeError('FAIL_CLOSED: feature artifact hash mismatch')
 # Explicit key ordering is used only after identity equality has been verified.
 Y=torch.tensor([teacher_by_key[k]['final_slots'][0] for k in keys],dtype=torch.float64)
 X=torch.tensor([feature_by_key[k]['feature'] for k in keys],dtype=torch.float64)
 # R2 EOS is recovered from the reconstruction bundle; use residual targets.
 bundle_path=Path.home()/'.cache/memorilla-lite/LITE-008V-R2/reconstruction.pt'
 if bundle_path.exists() and sha(bundle_path)!='73812ebd5810bee8115275f1dc6e291ace59895c41ff74951ebddb0721ac84fd': raise RuntimeError('FAIL_CLOSED: reconstruction bundle hash mismatch')
 bundle=torch.load(bundle_path,map_location='cpu') if bundle_path.exists() else None
 # R2 bundle location is authoritative in the R2 integrity report.
 if bundle is None:
  p=Path.home()/'.cache/memorilla-lite/LITE-008V-R2/reconstruction.pt'
  # no fallback to fabricated values; only emit algebraic shape/hash facts if bundle absent
  dump('LITE-008W-HISTORICAL-CROSS-VALIDATION.json',{'status':'BLOCKED','reason':'R2 reconstruction bundle path unavailable; no teacher-derived residual baseline loaded','source_feature_sha256':sha(F),'source_teacher_sha256':sha(R1),'folds':[]})
  return
 eos=bundle['eos'].double(); Y=Y-eos
 schedule=[0.0,1e-6,1e-3,1e-2,1e-1,1.0]
 folds=[]
 for entity in range(4):
  hold=[2*entity,2*entity+1]; train=[i for i in range(8) if i not in hold]
  row={'entity_index':entity,'held_out_indices':hold,'training_indices':train,'regularization':[]}
  for lam in schedule:
   Xt=X[train]; Yt=Y[train]
   if lam==0:
    A=torch.linalg.lstsq(Xt,Yt).solution
   else:
    A=torch.linalg.solve(Xt.T@Xt+lam*torch.eye(X.shape[1],dtype=torch.float64),Xt.T@Yt)
   pred=X@A
   row['regularization'].append({'lambda':lam,'fit':metrics(pred[train],Y[train]),'excluded':metrics(pred[hold],Y[hold]),'weight_norm':float(A.norm()),'held_out_reversal_evaluation':'BLOCKED_decoder_dependency'})
  folds.append(row)
 dump('LITE-008W-HISTORICAL-CROSS-VALIDATION.json',{'status':'COMPLETED_ALGEBRAIC_ONLY','scope':'historical diagnostic; not independent validation','folds':folds,'decoder_functional_metrics':'UNEXECUTED','optimizer_updates':0,'source_hashes':{'features':sha(F),'teachers':sha(R1)}})
 dump('LITE-008W-REGULARIZATION-ANALYSIS.json',{'status':'COMPLETED_ALGEBRAIC_ONLY','schedule':schedule,'selection_rule':'no independent development selection; report all values','note':'functional margin and reversal fields require frozen decoder and are blocked','optimizer_updates':0,'folds':folds})
 dump('LITE-008W-MATHEMATICAL-LEARNING-RESULTS.json',{'status':'COMPLETED_ALGEBRAIC_ONLY','model':'z=EOS+W*x','fit_source':'eight retained R2 historical teachers','optimizer_updates':0,'decoder_functional_qualification':'UNEXECUTED','projection_artifact':'outside_git_not_created_for_W_due_to_decoder_prerequisite_block'})
 dump('LITE-008W-QUERY-CONDITIONING-RESULTS.json',{'status':'UNEXECUTED','reason':'primary decoder-dependent and fresh-data prerequisites blocked','optimizer_updates':0})
 dump('LITE-008W-RESOURCE-TELEMETRY.json',{'status':'BLOCKED_BEFORE_W_EXECUTION','classification':'scientific_reference_evaluation_and_experimental_training_authorization','optimizer_updates':0,'algebraic_solver_operations':24,'decoder_forwards':0,'continuous_rss_monitor':'not used for algebraic-only diagnostic','limits':{'production_rss':268435456,'scientific_reference_rss':1073741824,'training_rss':1073741824,'min_available':536870912,'threads':1,'subprocesses':1,'max_updates':256,'max_wall_seconds':1800}})
 dump('LITE-008W-ARTIFACT-INTEGRITY.json',{'status':'verified_for_inputs','inputs':[{'path':str(R1),'sha256':sha(R1),'exists':True},{'path':str(F),'sha256':sha(F),'exists':True}], 'outside_git':True,'optimizer_updates':0,'new_teacher_artifacts':'none'})
 print('completed algebraic diagnostics')
if __name__=='__main__': main()
