"""Forward-only controls and fresh P oracle summary."""
import json,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import ROOT,MODEL,row_data,tokid,ids_for,init_slots,run_forward
CACHE=Path.home()/'.cache/memorilla-lite/LITE-008P'
def main():
 torch.set_num_threads(1);d=json.loads((ROOT/'dataset-manifest.json').read_text());t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();summ=[]
 for cid in ['case_08','case_09','case_10','case_11']:
  r=row_data(d,cid);ids=ids_for(t,r['query'],1);a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);e=m.get_input_embeddings();base=init_slots(m,t,r,1,'eos');zero=torch.zeros_like(base);rand=init_slots(m,t,r,1,'random');aa=e(torch.tensor([[a]])).detach().float();bb=e(torch.tensor([[b]])).detach().float();fr={}
  for name,s in [('initial_eos',base),('zero',zero),('scale_matched_random',rand),('genuine_answer_a',aa),('genuine_answer_b',bb)]:fr[name]=run_forward(m,ids,s,a,b)
  rs=json.loads((CACHE+Path(f'fresh-oracle-{cid[-2:]}') if False else CACHE/'fresh-oracle-08-09.json').read_text()) if cid in ['case_08','case_09'] else json.loads((CACHE/'fresh-oracle-10-11.json').read_text())
  ra=next(x for x in rs['results'] if x['case_id']==cid and x['target']=='A');rb=next(x for x in rs['results'] if x['case_id']==cid and x['target']=='B');fr['optimized_A']=run_forward(m,ids,torch.tensor(ra['final_slots']),a,b);fr['optimized_B']=run_forward(m,ids,torch.tensor(rb['final_slots']),a,b);summ.append({'case_id':cid,'initial_margin':fr['initial_eos']['margin'],'final_margin_M_A':fr['optimized_A']['margin'],'final_margin_M_B':fr['optimized_B']['margin'],'A_pass':fr['optimized_A']['margin']>0,'B_pass':fr['optimized_B']['margin']<0,'pair_pass':fr['optimized_A']['margin']>0 and fr['optimized_B']['margin']<0,'controls':fr,'trajectory_A':ra['trajectory'],'trajectory_B':rb['trajectory']})
 out={'schema':'LITE-008P-independent-oracle-v1','status':'completed','pairs':summ,'successful_A_directions':sum(x['A_pass'] for x in summ),'successful_B_directions':sum(x['B_pass'] for x in summ),'successful_counterfactual_pairs':sum(x['pair_pass'] for x in summ),'required_pairs':4,'required_successes':4,'protected_final_holdout_accessed':False};(ROOT/'oracle-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'A':out['successful_A_directions'],'B':out['successful_B_directions'],'pairs':out['successful_counterfactual_pairs']}))
if __name__=='__main__':main()
