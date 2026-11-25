"""LITE-008U mandatory training-set semantic reproduction gate; inference only."""
import copy,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import ids_for,run_forward,tokid
from lite.encoder.onnx_encoder import MiniLMEncoder
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();man=json.loads((REPO/'experiments/lite008q/dataset-manifest.json').read_text());rows={r['case_id']:r for r in man['rows']};state=json.loads((ROOT/'student-state.json').read_text());W=torch.tensor(state['W']);eos=torch.tensor(state['eos']);td=json.loads((ROOT/'teacher-dataset.json').read_text());teacher={(x['case_id'],x['target']):torch.tensor(x['embedding']) for x in td['teachers']};enc=MiniLMEncoder(threads=1);tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();results=[]
 for cid in sorted(rows):
  if rows[cid]['partition']!='train' or not any(x['case_id']==cid for x in td['teachers']):continue
  r=rows[cid];scores={}
  for target,text in [('A',r['evidence_a']),('B',r['evidence_b'])]:
   x=torch.from_numpy(enc.encode(text).astype('float32'));slot=eos+W@x;ids=ids_for(tok,r['query'],1);a=tokid(tok,r['answer_a']);b=tokid(tok,r['answer_b']);scores[target]=run_forward(m,ids,slot[None,:],a,b)|{'slot_norm':float(slot.norm()),'teacher_cosine':float(torch.nn.functional.cosine_similarity(slot[None],teacher[(cid,target)][None]).item())}
  okA=scores['A']['margin']>0;okB=scores['B']['margin']<0;results.append({'case_id':cid,'student_A':scores['A'],'student_B':scores['B'],'reversal':okA and okB,'A_correct':okA,'B_correct':okB})
 enc.close();success=sum(x['reversal'] for x in results);out={'schema':'LITE-008U-training-semantic-v1','status':'pass' if success/len(results)>=.75 else 'failed','qualified_pairs':len(results),'successful_reversals':success,'minimum_rate':.75,'rate':success/len(results),'rows':results,'wall_seconds':time.monotonic()-st,'independent_evaluation_authorized':success/len(results)>=.75,'decoder_refinement_authorized':success/len(results)>=.75};(ROOT/'training-semantic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'success':success,'pairs':len(results),'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
