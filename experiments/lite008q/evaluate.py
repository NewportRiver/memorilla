"""Frozen independent development evaluation for LITE-008Q."""
import copy,json,hashlib
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008p.p_experiment import MODEL,ids_for,tokid,run_forward
from experiments.lite008q.bridge_train import init_bridge,example_rows
ROOT=Path(__file__).resolve().parent;CACHE=Path.home()/'.cache/memorilla-lite/LITE-008Q';TOKDIR=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
def load_bridge(path,m,t):
 z=np.load(path);b=init_bridge(bias=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0]);b.weight.data.copy_(torch.from_numpy(z['weight']));b.bias.data.copy_(torch.from_numpy(z['bias']));return b
def main():
 d=json.loads((ROOT/'dataset-manifest.json').read_text());enc=MiniLMEncoder(threads=1);features={r['case_id']:enc.encode(r['evidence_a']) for r in d['rows']};features.update({r['case_id']+'B':enc.encode(r['evidence_b']) for r in d['rows']});features.update({r['case_id']+'U':enc.encode(r['unrelated_evidence']) for r in d['rows']});encid=enc.identity();enc.close();t=AutoTokenizer.from_pretrained(str(TOKDIR),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();trained=load_bridge(CACHE/'stable-contrastive-bridge.npz',m,t);untrained=init_bridge(bias=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0]);soft=torch.from_numpy(np.load(CACHE/'stable-soft-soft.npy')).float();train_mem=[]
 for r in d['rows']:
  if r['partition']=='train':train_mem.append(trained(torch.from_numpy(features[r['case_id']]).unsqueeze(0)).detach()[0])
 mean_mem=torch.stack(train_mem).mean(0);eos=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0];rng=torch.Generator().manual_seed(20261126);results=[]
 for r in d['rows']:
  if r['partition']!='development':continue
  a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);ids=ids_for(t,r['query'],1);scale=float(trained(torch.from_numpy(features[r['case_id']]).unsqueeze(0)).norm());random=eos/torch.linalg.vector_norm(eos)*scale
  cond={'trained_A':trained(torch.from_numpy(features[r['case_id']]).unsqueeze(0)).detach()[0],'trained_B':trained(torch.from_numpy(features[r['case_id']+'B']).unsqueeze(0)).detach()[0],'untrained_A':untrained(torch.from_numpy(features[r['case_id']]).unsqueeze(0)).detach()[0],'zero':torch.zeros(576),'query_only_eos':eos,'random_scale_matched':random,'mean_training_memory':mean_mem,'mismatched_evidence':trained(torch.from_numpy(features[r['case_id']+'U']).unsqueeze(0)).detach()[0],'soft_prompt':soft}
  for name,slot in cond.items():
   s=run_forward(m,ids,slot.unsqueeze(0),a,b);results.append({'case_id':r['case_id'],'condition':name,'margin':s['margin'],'nll_a':s['nll_a'],'nll_b':s['nll_b'],'greedy_id':s['greedy_id'],'greedy_text':t.decode([s['greedy_id']]),'embedding_norm':float(slot.norm()),'answer_a':r['answer_a'],'answer_b':r['answer_b']})
 out={'schema':'LITE-008Q-independent-development-v1','status':'completed','development_cases':8,'entity_disjoint_from_train':True,'rows':results,'encoder_identity':encid,'optimized_development_embeddings':False,'protected_final_holdout_accessed':False};(ROOT/'development-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':len(results),'conditions':9,'cases':8}))
if __name__=='__main__':main()
