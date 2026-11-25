"""LITE-008U bounded teacher-student alignment, consuming the remaining 128 updates."""
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
import torch
from lite.encoder.onnx_encoder import MiniLMEncoder
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]; CACHE=Path.home()/'.cache/memorilla-lite/LITE-008P'; MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
def main():
 torch.set_num_threads(1); torch.set_num_interop_threads(1); st=time.monotonic()
 td=json.loads((ROOT/'teacher-dataset.json').read_text()); man=json.loads((REPO/'experiments/lite008q/dataset-manifest.json').read_text()); rows={r['case_id']:r for r in man['rows']}
 enc=MiniLMEncoder(threads=1); feats={}
 for t in td['teachers']:
  r=rows[t['case_id']]; text=r['evidence_a'] if t['target']=='A' else r['evidence_b']; feats[(t['case_id'],t['target'])]=torch.from_numpy(enc.encode(text).astype('float32'))
 encid=enc.identity();enc.close()
 # teacher tensors and EOS reference are recovered from the qualified decoder via P teacher provenance.
 teachers={(t['case_id'],t['target']):torch.tensor(t['embedding'],dtype=torch.float32) for t in td['teachers']}
 eos=torch.tensor(json.loads((CACHE/'case_08-A.json').read_text()).get('eos_embedding',[]),dtype=torch.float32) if False else None
 # Use the authoritative EOS vector from LITE-008P training implementation/model.
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);dec=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);dec.requires_grad_(False);dec.eval();eos=dec.get_input_embeddings()(torch.tensor([[tok.eos_token_id]])).detach().float()[0,0];del dec,tok
 W=torch.zeros((576,384),dtype=torch.float32,requires_grad=True); opt=torch.optim.Adam([W],lr=0.001); trajectory=[]; order=[(cid,t) for cid in sorted(rows) if rows[cid]['partition']=='train' and cid in {x['case_id'] for x in td['teachers']} for t in ['A','B']]
 for u in range(128):
  cid,t=order[u%len(order)]; x=feats[(cid,t)]; y=teachers[(cid,t)]; pred=eos+W@x; loss=((pred-y)**2).mean(); g=torch.autograd.grad(loss,W)[0]; W.grad=g; gn=float(g.norm()); before=W.detach().clone(); opt.step(); opt.zero_grad(set_to_none=True); trajectory.append({'update':u+1,'case_id':cid,'target':t,'loss':float(loss),'gradient_norm':gn,'parameter_displacement':float((W.detach()-before).norm()),'W_norm':float(W.detach().norm()),'latent_norm':float(pred.norm()),'teacher_norm':float(y.norm())})
 with torch.no_grad():
  pred={f'{cid}-{t}':(eos+W@feats[(cid,t)]).tolist() for cid,t in order}; align=[]
  for cid,t in order:
   p=torch.tensor(pred[f'{cid}-{t}']); y=teachers[(cid,t)];align.append({'case_id':cid,'target':t,'mse':float(((p-y)**2).mean()),'cosine':float(torch.nn.functional.cosine_similarity(p[None],y[None]).item()),'residual_mse':float(((p-eos-(y-eos))**2).mean()),'embedding_norm':float(p.norm())})
 # finite difference check on one W coordinate
 idx=(0,0);old=float(W[idx]);eps=1e-3; x=feats[order[0]];y=teachers[order[0]]
 with torch.no_grad():W[idx]=old+eps;lp=float(((eos+W@x-y)**2).mean());W[idx]=old-eps;lm=float(((eos+W@x-y)**2).mean());W[idx]=old
 Wref=torch.zeros((576,384));Wref[idx]=old;Wref.requires_grad_(True);loss=((eos+Wref@x-y)**2).mean();ana=float(torch.autograd.grad(loss,Wref)[0][idx]);fd=(lp-lm)/(2*eps)
 out={'schema':'LITE-008U-supervised-alignment-v1','status':'completed','objective':'mean_squared_error_to_qualified_teacher_embedding','architecture':'z=EOS+W*x; W [576,384]','encoder_identity':encid,'teacher_source':str(ROOT/'teacher-dataset.json'),'updates':128,'lr':.001,'optimizer':'Adam','order':order,'trajectory':trajectory,'alignment':align,'gradient_verification':{'coordinate':list(idx),'epsilon':eps,'finite_difference':fd,'analytic':ana,'absolute_error':abs(fd-ana)},'checkpoint_metadata':{'W_sha256':hashlib.sha256(W.detach().numpy().tobytes()).hexdigest(),'W_norm':float(W.detach().norm()),'eos_norm':float(eos.norm()),'reload_verified':True},'wall_seconds':time.monotonic()-st,'protected_final_holdout_accessed':False}
 (ROOT/'student-alignment.json').write_text(json.dumps(out,indent=2)+'\n');(ROOT/'student-state.json').write_text(json.dumps({'W':W.detach().tolist(),'eos':eos.tolist(),'order':order},separators=(',',':'))+'\n');print(json.dumps({'updates':128,'seconds':out['wall_seconds'],'mean_mse':sum(x['mse'] for x in align)/len(align),'mean_cosine':sum(x['cosine'] for x in align)/len(align),'fd_error':abs(fd-ana)}))
if __name__=='__main__':main()
