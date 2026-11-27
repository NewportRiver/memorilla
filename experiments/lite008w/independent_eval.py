import json,time,hashlib,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import run_forward,tokid,ids_for,MODEL
ROOT=Path(__file__).resolve().parents[2];AUD=ROOT/'docs/memorilla-lite/audits';HOME=Path.home();CACHE=HOME/'.cache/memorilla-lite/LITE-008W-R3';MAN=AUD/'LITE-008W-R2-DATASET-MANIFEST.json';TD=CACHE/'teachers'
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic(); d=json.loads(MAN.read_text()); rows={x['id']:x for x in d['rows']}; textual=json.loads((AUD/'LITE-008W-R3-TEXTUAL-COMPETENCE-RESULTS.json').read_text()); textby={x['id']:x for x in textual['rows']}; eligible=['wtrain_00','wtrain_01','wtrain_02','wtrain_04','wtrain_05']; bundle=torch.load(CACHE/'projection_selected.pt',map_location='cpu'); W=bundle['W']; tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval(); eos=bundle['eos']; enc=MiniLMEncoder(threads=1); train_vec=[]; 
 for p in TD.glob('*.teacher.pt'):
  x=torch.load(p,map_location='cpu');
  if x['payload']['case_id'] in eligible: train_vec.append(x['embedding'].reshape(-1).float())
 mean=torch.stack(train_vec).mean(0); rng=torch.Generator().manual_seed(8008); rows_out=[]
 for r in [x for x in d['rows'] if x['partition']=='independent_development']:
  qids=ids_for(tok,r['query'],1); a=tokid(tok,r['answer_a']);b=tokid(tok,r['answer_b']); feats=[torch.from_numpy(enc.encode(r['evidence_a']).astype('float32')),torch.from_numpy(enc.encode(r['evidence_b']).astype('float32'))]; rand=torch.randn(eos.shape,generator=rng);rand=rand/rand.norm()*mean.norm(); candidates=[]
  for direction,feat in zip(['A','B'],feats):
   zsel=eos+feat@W.T; zmin=eos+feat@W.T; wrong=feats[1 if direction=='A' else 0]; zwrong=eos+wrong@W.T; zrandom=rand; methods={'minimum_norm':zmin,'selected_ridge':zsel,'query_only':eos,'zero_memory':eos,'scale_matched_random':zrandom,'mean_training_teacher':mean,'mismatched_evidence':zwrong,'untrained_projection':eos}
   scored={}
   for name,z in methods.items():
    o=run_forward(m,qids,z.reshape(1,-1),a,b);scored[name]={'margin':o['margin'],'nll_a':o['nll_a'],'nll_b':o['nll_b'],'greedy_id':o['greedy_id'],'latent_norm':float(z.norm()),'reversal':(o['margin']>0 if direction=='A' else o['margin']<0)}
   scored['ordinary_textual_evidence']=textby[r['id']]['relevant' if direction=='A' else 'counterfactual']; candidates.append({'direction':direction,'evidence':r['evidence_a' if direction=='A' else 'evidence_b'],'candidates':scored})
  rows_out.append({'id':r['id'],'entity':r['entity'],'textual_eligible':textby[r['id']]['eligible'],'directions':candidates})
 enc.close(); methods=['minimum_norm','selected_ridge','query_only','zero_memory','scale_matched_random','mean_training_teacher','mismatched_evidence','untrained_projection']; summary={name:{'reversals':sum(row['directions'][0]['candidates'][name]['reversal'] and row['directions'][1]['candidates'][name]['reversal'] for row in rows_out),'pairs':len(rows_out)} for name in methods}; out={'status':'COMPLETED_CONTROLS_INCOMPLETE','rows':rows_out,'summary':summary,'soft_prompt':'UNEXECUTED; remaining budget 30 < required 32','selection_frozen_before_development':True,'optimizer_updates':0,'wall_seconds':time.monotonic()-st,'protected_final_holdout_accessed':False}; (AUD/'LITE-008W-R3-INDEPENDENT-DEVELOPMENT-RESULTS.json').write_text(json.dumps(out,indent=2)+'\n'); (AUD/'LITE-008W-R3-CONTROL-COMPARISON.json').write_text(json.dumps({'status':'COMPLETED_EXCEPT_SOFT_PROMPT','summary':summary,'soft_prompt':'BLOCKED_BY_CUMULATIVE_BUDGET','rows':rows_out},indent=2)+'\n'); print(json.dumps({'summary':summary,'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
