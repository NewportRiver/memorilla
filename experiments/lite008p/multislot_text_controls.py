"""P numerical multi-slot and exact textual embedding controls."""
import copy,json,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import MODEL,ROOT,ids_for,tokid,masks,layer_call,fp_components,run_forward,objective

def arbitrary_forward(m,ids,emb):
 x=emb.detach().float();posids,pos,mask,pe=masks(m,x);h=x
 for i in range(30):
  l=copy.deepcopy(m.model.layers[i]).float()
  with torch.no_grad():h=layer_call(l,h,mask,posids,pos,pe).detach()
  del l
 norm,ow=fp_components(m);return torch.nn.functional.linear(norm(h)[:,-1,:],ow).float()
def main():
 torch.set_num_threads(1);st=time.monotonic();d=json.loads((ROOT/'dataset-manifest.json').read_text());r=next(x for x in d['rows'] if x['case_id']=='case_00');t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);n=2;ids=ids_for(t,r['query'],n);slots=m.get_input_embeddings()(ids).detach().float()[:,:n,:][0];x=slots.clone();x[0]+=torch.randn_like(x[0])*0.01;obj=objective(m,ids,x,a,b,True,'contrastive');g=obj['grad'];v=torch.randn_like(x);v/=v.norm();eps=.01
 def scalar(s):
  e=run_forward(m,ids,s,a,b);return e['nll_a']+.5*float(torch.nn.functional.softplus(torch.tensor(-e['margin'])))
 fd=(scalar(x+eps*v)-scalar(x-eps*v))/(2*eps);ad=float((g*v).sum());
 prompt=r['query']+' '+r['relevant_evidence']+' Answer:';full=torch.tensor([t(prompt,add_special_tokens=True)['input_ids']]);std=m(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False).logits.float()[:,-1,:];emb=m.get_input_embeddings()(full).detach().float();inj=arbitrary_forward(m,full,emb);diff=float((std-inj).abs().max());
 ev={'schema':'LITE-008P-multislot-text-controls-v1','multislot_gradient_norms':[float(z) for z in g.norm(dim=1)],'finite_difference_directional_derivative':fd,'analytic_directional_derivative':ad,'finite_difference_relative_error':abs(fd-ad)/max(1e-6,abs(fd),abs(ad)),'text_exact_embedding_max_logit_abs_diff':diff,'text_candidate_margin_standard':float(std[0,a]-std[0,b]),'text_candidate_margin_injected':float(inj[0,a]-inj[0,b]),'slot_count':2,'wall_seconds':time.monotonic()-st,'decoder_parameters_frozen':True,'protected_final_holdout_accessed':False};(ROOT/'multislot-text-controls.json').write_text(json.dumps(ev,indent=2)+'\n');print(json.dumps(ev))
if __name__=='__main__':main()
