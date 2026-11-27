import argparse,json,time,hashlib,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import run_forward,tokid,ids_for,MODEL
ROOT=Path(__file__).resolve().parents[1];AUD=ROOT/'docs/memorilla-lite/audits';HOME=Path.home();MAN=AUD/'LITE-008X-DATASET-MANIFEST.json';FEAT=HOME/'.cache/memorilla-lite/LITE-008W-R2/features.json';R3=HOME/'.cache/memorilla-lite/LITE-008W-R3';R4=HOME/'.cache/memorilla-lite/LITE-008W-R4';ELIG=['wtrain_00','wtrain_01','wtrain_02','wtrain_04','wtrain_05']
def sha(b):return hashlib.sha256(b).hexdigest()
def main(group):
 torch.set_num_threads(1);torch.set_num_interop_threads(1); st=time.monotonic();D=json.loads(MAN.read_text()); rows=[r for r in D['rows'] if r['group']==group]; fd=json.loads(FEAT.read_text()); feats={(x['id'],x['target']):torch.tensor(x['feature'],dtype=torch.float64) for x in fd['features']}; teachers=[]
 for p in (R3/'teachers').glob('*.teacher.pt'):
  x=torch.load(p,map_location='cpu');
  if x['payload']['case_id'] in ELIG: teachers.append(x)
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();eos=m.get_input_embeddings()(torch.tensor([[tok.eos_token_id]])).detach().float()[0,0];X=torch.stack([feats[(x['payload']['case_id'],x['payload']['target'])] for x in teachers]);Y=torch.stack([x['embedding'].reshape(-1).double()-eos.double() for x in teachers]);W0=torch.linalg.lstsq(X,Y).solution.T.float();W1=torch.linalg.solve(X.T@X+1e-3*torch.eye(X.shape[1],dtype=torch.float64),X.T@Y).T.float();mean=torch.stack([x['embedding'].reshape(-1).float() for x in teachers]).mean(0);soft=torch.load(next(R4.glob('*.soft_prompt.pt')),map_location='cpu')['prompt'].reshape(-1);enc=MiniLMEncoder(threads=1);gen=torch.Generator().manual_seed(8008);outrows=[];names=['ridge','minimum_norm','query_eos','zero_embedding','untrained_residual','random','mean_teacher','soft_prompt','opposite_fact','unrelated_entity','answer_value_only','entity_only','relation_swapped','evidence_order_permuted']
 for r in rows:
  a,b=r['answer_a'],r['answer_b'];qids=ids_for(tok,r['query'],1);ta=tokid(tok,a);tb=tokid(tok,b);F=[torch.from_numpy(enc.encode(r['evidence_a']).astype('float32')),torch.from_numpy(enc.encode(r['evidence_b']).astype('float32'))]; unrelated=torch.from_numpy(enc.encode(r['unrelated_evidence']).astype('float32')); vals=[]
  for j,direction in enumerate(['A','B']):
   textans=a if direction=='A' else b; entity=torch.from_numpy(enc.encode(f'Record {r["entity"]}.').astype('float32')); ansf=torch.from_numpy(enc.encode(f'value {textans}.').astype('float32')); ordertext=r['evidence_a' if direction=='A' else 'evidence_b'];
   if group=='D':
    parts=ordertext.split('; '); ordertext='; '.join(reversed(parts))
   orderf=torch.from_numpy(enc.encode(ordertext).astype('float32')); wrongF=F[1-j]; rand=torch.randn(eos.shape,generator=gen);rand=rand/rand.norm()*mean.norm(); z={'ridge':eos+F[j]@W1.T,'minimum_norm':eos+F[j]@W0.T,'query_eos':eos,'zero_embedding':torch.zeros_like(eos),'untrained_residual':eos,'random':rand,'mean_teacher':mean,'soft_prompt':soft,'opposite_fact':eos+wrongF@W1.T,'unrelated_entity':eos+unrelated@W1.T,'answer_value_only':eos+ansf@W1.T,'entity_only':eos+entity@W1.T,'relation_swapped':eos+wrongF@W1.T,'evidence_order_permuted':eos+orderf@W1.T}; scores={}
   for n in names:
    t0=time.monotonic();o=run_forward(m,qids,z[n].reshape(1,-1),ta,tb);scores[n]={'margin':o['margin'],'nll_a':o['nll_a'],'nll_b':o['nll_b'],'greedy_id':o['greedy_id'],'latent_norm':float(z[n].norm()),'reversal':bool(o['margin']>0 if direction=='A' else o['margin']<0),'latency_seconds':time.monotonic()-t0}
   vals.append({'direction':direction,'evidence_hash':sha(r['evidence_a' if direction=='A' else 'evidence_b'].encode()),'scores':scores})
  outrows.append({'id':r['id'],'directions':vals})
 enc.close(); summary={n:{'reversals':sum(x['directions'][0]['scores'][n]['reversal'] and x['directions'][1]['scores'][n]['reversal'] for x in outrows),'pairs':len(outrows)} for n in names};out={'status':'FORWARD_ONLY_EXPLORATORY','group':group,'rows':outrows,'summary':summary,'optimizer_updates':0,'candidate_independent_representation':True,'wall_seconds':time.monotonic()-st,'protected_final_holdout_accessed':False};(AUD/f'LITE-008X-{group}-LATENT-EVALUATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'group':group,'summary':summary,'seconds':out['wall_seconds']}))
if __name__=='__main__':
 p=__import__('argparse').ArgumentParser();p.add_argument('--group',required=True);main(p.parse_args().group)
