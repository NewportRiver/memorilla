import json,time,hashlib,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import run_forward,tokid,ids_for,MODEL
ROOT=Path(__file__).resolve().parents[2];AUD=ROOT/'docs/memorilla-lite/audits';HOME=Path.home();R3=HOME/'.cache/memorilla-lite/LITE-008W-R3';MAN=AUD/'LITE-008W-R2-DATASET-MANIFEST.json'
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1); st=time.monotonic(); D=json.loads(MAN.read_text()); dev=[r for r in D['rows'] if r['partition']=='independent_development']; tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval(); eos=m.get_input_embeddings()(torch.tensor([[tok.eos_token_id]])).detach().float()[0,0]; proj=torch.load(R3/'projection_selected.pt',map_location='cpu');W=proj['W']; enc=MiniLMEncoder(threads=1); rows=[]
 for r in dev:
  a=tokid(tok,r['answer_a']);b=tokid(tok,r['answer_b']); qids=ids_for(tok,r['query'],1); out={'id':r['id'],'directions':[]};
  for direction,ans,other in [('A',r['answer_a'],r['answer_b']),('B',r['answer_b'],r['answer_a'])]:
   ev=r['evidence_a' if direction=='A' else 'evidence_b']; variants={'full_relevant':ev,'answer_value_only':f'value {ans}.','entity_only':f'Record {r["entity"]}.','same_answer_other_entity':f'Record unrelated stores value {ans}.','opposite_factual_value':f'Record {r["entity"]} stores value {other}.','meaning_preserving_paraphrase':f'The value recorded for {r["entity"]} is {ans}.','conflicting_evidence':f'Record {r["entity"]} stores value {ans}, not {other}.'}; scores={}
   for name,text in variants.items():
    f=torch.from_numpy(enc.encode(text).astype('float32'));z=eos+f@W.T;o=run_forward(m,qids,z.reshape(1,-1),a,b);scores[name]={'margin':o['margin'],'nll_a':o['nll_a'],'nll_b':o['nll_b'],'greedy_id':o['greedy_id'],'latent_norm':float(z.norm()),'reversal':bool(o['margin']>0 if direction=='A' else o['margin']<0),'input_hash':hashlib.sha256(text.encode()).hexdigest()}
   out['directions'].append({'direction':direction,'scores':scores})
  rows.append(out)
 enc.close(); names=list(rows[0]['directions'][0]['scores']); summary={n:{'reversals':sum(x['directions'][0]['scores'][n]['reversal'] and x['directions'][1]['scores'][n]['reversal'] for x in rows),'pairs':len(rows)} for n in names}; out={'status':'EXPLORATORY_FORWARD_ONLY','selection_or_training_used':False,'rows':rows,'summary':summary,'wall_seconds':time.monotonic()-st,'protected_final_holdout_accessed':False}; (AUD/'LITE-008W-R4-SEMANTIC-MECHANISM-DIAGNOSTICS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'summary':summary,'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
