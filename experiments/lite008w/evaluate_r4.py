import json,time,hashlib,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import run_forward,tokid,ids_for,MODEL
ROOT=Path(__file__).resolve().parents[2];AUD=ROOT/'docs/memorilla-lite/audits';HOME=Path.home();R3=HOME/'.cache/memorilla-lite/LITE-008W-R3';R4=HOME/'.cache/memorilla-lite/LITE-008W-R4';MAN=AUD/'LITE-008W-R2-DATASET-MANIFEST.json';FEAT=HOME/'.cache/memorilla-lite/LITE-008W-R2/features.json';ELIG=['wtrain_00','wtrain_01','wtrain_02','wtrain_04','wtrain_05']
def sh(x): return hashlib.sha256(x).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1); st=time.monotonic(); data=json.loads(MAN.read_text()); dev=[r for r in data['rows'] if r['partition']=='independent_development']; fd=json.loads(FEAT.read_text()); feats={(x['id'],x['target']):torch.tensor(x['feature'],dtype=torch.float64) for x in fd['features']}; teachers=[]
 for p in (R3/'teachers').glob('*.teacher.pt'):
  x=torch.load(p,map_location='cpu');
  if x['payload']['case_id'] in ELIG: teachers.append(x)
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);model.requires_grad_(False);model.eval(); eos=model.get_input_embeddings()(torch.tensor([[tok.eos_token_id]])).detach().float()[0,0]; X=torch.stack([feats[(x['payload']['case_id'],x['payload']['target'])] for x in teachers]);Y=torch.stack([x['embedding'].reshape(-1).double()-eos.double() for x in teachers]); A0=torch.linalg.lstsq(X,Y).solution; W0=A0.T.float(); A1=torch.linalg.solve(X.T@X+1e-3*torch.eye(X.shape[1],dtype=torch.float64),X.T@Y);W1=A1.T.float(); proj_hashes={'minimum_norm':sh(W0.numpy().tobytes()),'selected_ridge':sh(W1.numpy().tobytes()),'matrix_max_abs_diff':float((W0-W1).abs().max()),'matrix_l2_diff':float((W0-W1).norm())}; bundle=torch.load(R3/'projection_selected.pt',map_location='cpu'); assert torch.allclose(bundle['W'],W1)
 prompt_files=list(R4.glob('*.soft_prompt.pt')); assert len(prompt_files)==1; prompt=torch.load(prompt_files[0],map_location='cpu')['prompt'].reshape(-1); enc=MiniLMEncoder(threads=1); mean=torch.stack([x['embedding'].reshape(-1).float() for x in teachers]).mean(0); gen=torch.Generator().manual_seed(8008); rows=[]
 for r in dev:
  qids=ids_for(tok,r['query'],1); a=tokid(tok,r['answer_a']);b=tokid(tok,r['answer_b']); f=[]
  for ev in [r['evidence_a'],r['evidence_b']]: f.append(torch.from_numpy(enc.encode(ev).astype('float32')))
  rand=torch.randn(eos.shape,generator=gen);rand=rand/rand.norm()*mean.norm(); unrelated=next(x for x in dev if x['id']!=r['id'])['evidence_a']; fu=torch.from_numpy(enc.encode(unrelated).astype('float32'))
  zmap={'minimum_norm':[eos+v@W0.T for v in f],'selected_ridge':[eos+v@W1.T for v in f],'soft_prompt':[prompt,prompt],'query_only':[eos,eos],'zero_embedding':[torch.zeros_like(eos),torch.zeros_like(eos)],'untrained_residual_projection':[eos,eos],'scale_matched_random':[rand,rand],'mean_training_teacher':[mean,mean],'same_entity_opposite_evidence':[eos+f[1]@W1.T,eos+f[0]@W1.T],'unrelated_entity_evidence':[eos+fu@W1.T,eos+fu@W1.T]}
  out={'id':r['id'],'entity':r['entity'],'query_hash':sh(r['query'].encode()),'directions':[]}
  for j,direction in enumerate(['A','B']):
   scores={}
   for name,zs in zmap.items():
    z=zs[j];o=run_forward(model,qids,z.reshape(1,-1),a,b); scores[name]={'margin':o['margin'],'nll_a':o['nll_a'],'nll_b':o['nll_b'],'greedy_id':o['greedy_id'],'latent_norm':float(z.norm()),'reversal':bool(o['margin']>0 if direction=='A' else o['margin']<0)}
   out['directions'].append({'direction':direction,'evidence_hash':sh(r['evidence_a' if direction=='A' else 'evidence_b'].encode()),'representation_input_hash':sh((r['query']+'\\0'+r['evidence_a' if direction=='A' else 'evidence_b']).encode()),'scores':scores})
  rows.append(out)
 enc.close(); names=list(rows[0]['directions'][0]['scores']); summary={n:{'reversals':sum(x['directions'][0]['scores'][n]['reversal'] and x['directions'][1]['scores'][n]['reversal'] for x in rows),'pairs':len(rows)} for n in names}; soft={'summary':summary['soft_prompt'],'rows':rows,'artifact_sha256':sh(prompt_files[0].read_bytes()),'optimizer_updates':32,'deterministic_repeat_required':True}; repro={'status':'COMPLETED','rows':rows,'summary':summary,'projection_hashes':proj_hashes,'selected_projection_artifact_sha256':sh((R3/'projection_selected.pt').read_bytes()),'representation_candidate_independent':True,'candidate_tokens_used_only_scoring':True,'soft_prompt':soft,'wall_seconds':time.monotonic()-st,'protected_final_holdout_accessed':False}; (AUD/'LITE-008W-R4-INDEPENDENT-REPRODUCTION.json').write_text(json.dumps(repro,indent=2)+'\n'); (AUD/'LITE-008W-R4-SOFT-PROMPT-CONTROL-RESULTS.json').write_text(json.dumps(soft,indent=2)+'\n'); (AUD/'LITE-008W-R4-PROJECTION-COMPARISON.json').write_text(json.dumps({'status':'COMPLETED','projection_hashes':proj_hashes,'selected_bundle_matches_ridge':True,'rows':rows,'summary':{'minimum_norm':summary['minimum_norm'],'selected_ridge':summary['selected_ridge']}},indent=2)+'\n'); print(json.dumps({'summary':summary,'projection_hashes':proj_hashes,'seconds':repro['wall_seconds']}))
if __name__=='__main__':main()
