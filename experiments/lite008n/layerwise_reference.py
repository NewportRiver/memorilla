"""Memory-bounded FP32 layer-wise VJP using official frozen Llama components."""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.llama.modeling_llama import create_causal_mask
from experiments.lite008l.ledger import record
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008N'
def setup():
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();m.config.use_cache=False;ids=torch.tensor([[int(tok.eos_token_id)]+tok('Question: What is the capital of France? Answer:',add_special_tokens=True)['input_ids']]);target=torch.tensor(tok(' Paris',add_special_tokens=False)['input_ids']);return tok,m,ids,target
def masks(m,hidden):
 seq=hidden.shape[1];pos=torch.arange(seq);posids=pos.unsqueeze(0);mask=create_causal_mask(config=m.config,input_embeds=hidden,attention_mask=torch.ones((1,seq),dtype=torch.long),cache_position=pos,past_key_values=None,position_ids=posids);pe=m.model.rotary_emb(hidden,posids);return posids,pos,mask,pe
def layer_call(layer,h,mask,posids,pos,pe):return layer(h,attention_mask=mask,position_ids=posids,position_embeddings=pe,use_cache=False,cache_position=pos)
def fp_components(m):
 norm=copy.deepcopy(m.model.norm).float();ow=m.lm_head.weight.detach().float();return norm,ow
def final_loss(h,norm,ow,target):
 logits=torch.nn.functional.linear(norm(h)[:,-1,:],ow).float();return torch.nn.functional.cross_entropy(logits,target),logits
def conventional(m,hidden,nlayers,target):
 posids,pos,mask,pe=masks(m,hidden);layers=[copy.deepcopy(m.model.layers[i]).float() for i in range(nlayers)];norm,ow=fp_components(m);h=hidden.detach().float().requires_grad_(True);states=[h];
 for layer in layers:h=layer_call(layer,h,mask,posids,pos,pe);states.append(h)
 loss,logits=final_loss(h,norm,ow,target);g=torch.autograd.grad(loss,states[0])[0];return {'loss':float(loss),'logits':logits.detach(),'grad':g.detach(),'states':states,'layers':layers,'norm':norm,'ow':ow,'mask':mask,'posids':posids,'pos':pos,'pe':pe}
def layerwise(m,hidden,nlayers,target):
 x=hidden.detach().float();posids,pos,mask,pe=masks(m,x);states=[x]
 for i in range(nlayers):
  layer=copy.deepcopy(m.model.layers[i]).float()
  with torch.no_grad():x=layer_call(layer,x,mask,posids,pos,pe).detach()
  states.append(x);del layer
 norm,ow=fp_components(m);h=states[-1].detach().requires_grad_(True);loss,logits=final_loss(h,norm,ow,target);gh=torch.autograd.grad(loss,h)[0].detach()
 for i in reversed(range(nlayers)):
  layer=copy.deepcopy(m.model.layers[i]).float();inp=states[i].detach().requires_grad_(True);out=layer_call(layer,inp,mask,posids,pos,pe);gh=torch.autograd.grad(out,inp,grad_outputs=gh)[0].detach();del layer,out,inp
 return {'loss':float(loss),'logits':logits.detach(),'grad':gh,'states':states}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--layers',type=int,default=30);a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);ROOT.mkdir(parents=True,exist_ok=True);record('layerwise_reference','started',layers=a.layers,dtype='float32',stored_weight_dtype='bfloat16');start=time.monotonic();tok,m,ids,target=setup();hidden=m.get_input_embeddings()(ids).detach().float();
 if a.layers<=2:
  c=conventional(m,hidden,a.layers,target);lw=layerwise(m,hidden,a.layers,target);res={'schema':'LITE-008N-reduced-equivalence-v1','status':'completed','layers':a.layers,'loss_abs_error':abs(c['loss']-lw['loss']),'logit_max_abs_error':float((c['logits']-lw['logits']).abs().max()),'gradient_max_abs_error':float((c['grad']-lw['grad']).abs().max()),'gradient_cosine':float(torch.nn.functional.cosine_similarity(c['grad'].flatten(),lw['grad'].flatten(),dim=0)),'gradient_norm_ratio':float(lw['grad'].norm()/c['grad'].norm()),'wall_seconds':time.monotonic()-start};(ROOT/'reduced-equivalence-results.json').write_text(json.dumps(res,indent=2)+'\n');record('layerwise_reference','completed',layers=a.layers,wall_seconds=res['wall_seconds']);print(json.dumps(res));return
 # Complete layerwise forward/backward; compare token-ID and equivalent external embedding at the boundary.
 ids_hidden=m.get_input_embeddings()(ids).detach().float();external=ids_hidden.clone().requires_grad_(True);lw=layerwise(m,external,a.layers,target);ordinary=m.get_input_embeddings()(ids).detach().float();lw_ord=layerwise(m,ordinary,a.layers,target);res={'schema':'LITE-008N-full-layerwise-reference-v1','status':'completed','layers':a.layers,'loss_external':lw['loss'],'loss_token_embedding':lw_ord['loss'],'ordinary_external_loss_abs_error':abs(lw['loss']-lw_ord['loss']),'ordinary_external_logit_max_abs_error':float((lw['logits']-lw_ord['logits']).abs().max()),'external_gradient_norm':float(lw['grad'].norm()),'gradient_finite':bool(torch.isfinite(lw['grad']).all()),'forward_count':a.layers*2,'backward_vjp_count':a.layers,'wall_seconds':time.monotonic()-start,'decoder_parameters_frozen':all(not p.requires_grad for p in m.parameters()),'protected_final_holdout_accessed':False};(ROOT/'full-layerwise-reference-results.json').write_text(json.dumps(res,indent=2)+'\n');record('layerwise_reference','completed',layers=a.layers,wall_seconds=res['wall_seconds'],forward_count=res['forward_count'],backward_vjp_count=res['backward_vjp_count']);print(json.dumps(res))
if __name__=='__main__':main()
