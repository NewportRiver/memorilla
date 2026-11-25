"""LITE-008V-R2 zero-update reference evaluation and reconstruction."""
from __future__ import annotations
import copy, hashlib, json, math, os, statistics, time
from pathlib import Path
import numpy as np
import torch
from lite.encoder.onnx_encoder import MiniLMEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.lite008p.p_experiment import run_forward, tokid, ids_for, MODEL

ROOT=Path(__file__).resolve().parents[2]; AUD=ROOT/'docs/memorilla-lite/audits'; CACHE=Path.home()/'.cache/memorilla-lite/LITE-008V-R2'; CACHE.mkdir(parents=True,exist_ok=True)

def dh(v): return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
def fh(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def rss():
 try:return int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
 except:return None
def avail():
 try:return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
 except:return None
def write(name,obj): (AUD/name).write_text(json.dumps(obj,indent=2)+'\n')
def scores(m,t,r,slot):
 a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);out=run_forward(m,ids_for(t,r['query'],1),slot.reshape(1,-1),a,b);return out

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic()
 pm=json.loads((ROOT/'experiments/lite008p/dataset-manifest.json').read_text());rows={r['case_id']:r for r in pm['rows']}; td=json.loads((Path.home()/'.cache/memorilla-lite/LITE-008P/lite008u-teachers.json').read_text())
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval()
 eos=m.get_input_embeddings()(torch.tensor([[tok.eos_token_id]])).detach().float()[0,0]
 enc=MiniLMEncoder(threads=1);encid=enc.identity(); records=[];features=[];teachers=[]
 for z in td['results']:
  r=rows[z['case_id']]; vec=torch.tensor(z['final_slots'][0],dtype=torch.float32); text=r['relevant_evidence'] if z['target']=='A' else r['counterfactual_evidence']; feat=torch.from_numpy(enc.encode(text).astype(np.float32));
  out=scores(m,tok,r,vec); expected=z['trajectory'][-1]['margin'];
  records.append({'case_id':z['case_id'],'target':z['target'],'query':r['query'],'candidate_answers':[r['answer_a'],r['answer_b']],'candidate_token_ids':[tokid(tok,r['answer_a']),tokid(tok,r['answer_b'])],'evidence':text,'historical_margin':expected,'reloaded_margin':out['margin'],'margin_delta':out['margin']-expected,'nll_a':out['nll_a'],'nll_b':out['nll_b'],'norm':float(vec.norm()),'preference':('A' if out['margin']>0 else 'B'),'historical_preference':z['target'],'vector_sha256':dh(z['final_slots']),'feature_sha256':dh(feat.tolist()),'feature_norm':float(feat.norm()),'feature_dim':int(feat.numel()),'encoder_identity':encid});teachers.append(vec);features.append(feat)
 enc.close();
 write('LITE-008V-R2-TEACHER-REQUALIFICATION.json',{'status':'PASS' if all(x['preference']==x['historical_preference'] for x in records) else 'FAIL','tolerance_margin_abs':1e-4,'records':records,'optimizer_updates':0,'teacher_regeneration':False})
 if not all(x['preference']==x['historical_preference'] for x in records): raise RuntimeError('teacher requalification failed')
 X=torch.stack(features).double();Y=torch.stack([v-eos.double() for v in teachers]);u,s,vh=torch.linalg.svd(X,full_matrices=False);rank=int((s>max(X.shape)*torch.finfo(torch.float64).eps*s[0]).sum());cond=float(s[0]/s[-1]) if s[-1]>0 else float('inf');
 A=torch.linalg.lstsq(X,Y).solution; W=A.T; pred=X@A; ridge_lam=1e-6; At=torch.linalg.solve(X.T@X+ridge_lam*torch.eye(X.shape[1],dtype=torch.float64),X.T@Y);Wr=At.T
 def matmetrics(Z):
  E=Z-Y; return {'mse':float((E*E).mean()),'max_abs':float(E.abs().max()),'relative_l2':float(E.norm()/Y.norm()),'residual_mse':float((E*E).mean()),'output_norm_range':[float((eos.double()+z).norm()) for z in Z]}
 write('LITE-008V-R2-FEATURE-MATRIX-ANALYSIS.json',{'shape':list(X.shape),'rank':rank,'singular_values':s.tolist(),'condition_number':cond,'feature_norms':[float(x.norm()) for x in X],'feature_pair_distances':[float((X[i]-X[i+1]).norm()) for i in range(0,8,2)],'feature_dtypes':['float32_source','float64_solver'],'encoder_identity':encid,'counterfactual_feature_separation':[(records[i]['case_id'],float((X[i]-X[i+1]).norm())) for i in range(0,8,2)]})
 W32=W.float();Wr32=Wr.float(); pred32=X.float()@W32.T;predr32=X.float()@Wr32.T
 write('LITE-008V-R2-EXACT-RECONSTRUCTION-RESULTS.json',{'float64_lstsq':{**matmetrics(pred), 'weight_norm':float(W.norm())},'float64_pseudoinverse':{**matmetrics(X@torch.linalg.pinv(X)@Y),'weight_norm':float((torch.linalg.pinv(X)@Y).T.norm())},'float64_ridge':{**matmetrics(X@At),'weight_norm':float(Wr.norm()),'lambda':ridge_lam},'float32_lstsq_after_conversion':{**matmetrics(pred32.double()),'weight_norm':float(W32.norm())},'float32_ridge_after_conversion':{**matmetrics(predr32.double()),'weight_norm':float(Wr32.norm())},'optimizer_updates':0,'artifact_hashes':{'W_float64':dh(W.tolist()),'W_float32':dh(W32.tolist()),'W_ridge_float32':dh(Wr32.tolist())}})
 torch.save({'W':W32,'W_ridge':Wr32,'eos':eos,'feature_hashes':[x['feature_sha256'] for x in records]},CACHE/'reconstruction.pt');
 cases=[(rows[z['case_id']],z['target'],torch.tensor(z['final_slots'][0],dtype=torch.float32)) for z in td['results']]; rng=torch.Generator().manual_seed(8008); randoms=[torch.randn(eos.shape,generator=rng) for _ in cases]; randoms=[x/x.norm()*eos.norm() for x in randoms]
 evalrows=[]
 for r,target,teacher in cases:
  i=next(i for i,z in enumerate(td['results']) if z['case_id']==r['case_id'] and z['target']==target); exact=pred32[i]+eos; ridge=predr32[i]+eos; mean=torch.stack(teachers).mean(0); opts=[('teacher',teacher),('lstsq',exact),('ridge',ridge),('eos',eos),('mean',mean),('random',randoms[i])]; cand=[]
  for name,z in opts:
   o=scores(m,tok,r,z);cand.append({'name':name,**{k:o[k] for k in ('margin','nll_a','nll_b','greedy_id')},'norm':float(z.norm())})
  evalrows.append({'case_id':r['case_id'],'target':target,'candidate_answers':[r['answer_a'],r['answer_b']],'candidates':cand})
 write('LITE-008V-R2-FUNCTIONAL-EQUIVALENCE-RESULTS.json',{'rows':evalrows,'reversal_counts':{name:sum(next(x for x in row['candidates'] if x['name']==name)['margin']>0 if row['target']=='A' else next(x for x in row['candidates'] if x['name']==name)['margin']<0 for row in evalrows) for name in ['teacher','lstsq','ridge','eos','mean','random']},'optimizer_updates':0})
 sens=[]
 for r,target,teacher in cases:
  i=next(i for i,z in enumerate(td['results']) if z['case_id']==r['case_id'] and z['target']==target); exact=pred32[i]+eos; direction=torch.arange(1,577,dtype=torch.float32);direction=direction/direction.norm(); row={'case_id':r['case_id'],'target':target,'teacher_to_reconstruction':[],'teacher_to_eos':[],'matched_perturbations':[]}
  for q in [0,.25,.5,.75,1]:
   for key,z in [('teacher_to_reconstruction',teacher*(1-q)+exact*q),('teacher_to_eos',teacher*(1-q)+eos*q)]: row[key].append({'alpha':q,**{k:scores(m,tok,r,z)[k] for k in ('margin','nll_a','nll_b')}})
  for scale in [.01,.1,1.0]:
   z=teacher+direction*scale;row['matched_perturbations'].append({'euclidean_scale':scale,**{k:scores(m,tok,r,z)[k] for k in ('margin','nll_a','nll_b')}})
  sens.append(row)
 write('LITE-008V-R2-DECODER-SENSITIVITY-RESULTS.json',{'interpolation_schedule':[0,.25,.5,.75,1],'perturbation_scales':[.01,.1,1.0],'rows':sens,'optimizer_updates':0})
 write('LITE-008V-R2-FEATURE-ARTIFACT-MANIFEST.json',{'status':'recoverable','artifact_path':str(CACHE/'features.json'),'teacher_feature_count':8,'feature_dim':384,'encoder_identity':encid,'source':'P manifest evidence; not teacher-generation input','optimizer_updates':0})
 (CACHE/'features.json').write_text(json.dumps({'encoder_identity':encid,'features':[{'case_id':x['case_id'],'target':x['target'],'feature_sha256':x['feature_sha256'],'feature':features[i].tolist()} for i,x in enumerate(records)]},indent=2)+'\n')
 write('LITE-008V-R2-RESOURCE-TELEMETRY.json',{'status':'completed','classification':'scientific_reference_evaluation','optimizer_updates':0,'solver_operations':2,'decoder_forwards':len(records)+len(evalrows)*6+len(sens)*13,'wall_seconds':time.monotonic()-st,'peak_rss_bytes':rss(),'minimum_available_bytes':avail(),'rss_ceiling_bytes':1073741824,'production_rss_ceiling_bytes':268435456})
 print(json.dumps({'seconds':time.monotonic()-st,'rss':rss(),'reloaded':8,'reversals':len(evalrows)}))
if __name__=='__main__':main()
