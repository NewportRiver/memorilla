"""Reduced two-layer FP32 MiniLM-feature -> bridge -> decoder gradient-chain check."""
import json,torch,numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008n.layerwise_reference import masks,layer_call,fp_components,layerwise
from experiments.lite008q.bridge_train import init_bridge
ROOT=Path(__file__).resolve().parent;MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';TOKDIR=MODEL
d=json.loads((ROOT/'dataset-manifest.json').read_text());r=next(x for x in d['rows'] if x['partition']=='train');enc=MiniLMEncoder(threads=1);x=torch.from_numpy(enc.encode(r['evidence_a']));encid=enc.identity();enc.close();t=AutoTokenizer.from_pretrained(str(TOKDIR),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();eos=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0];b=init_bridge(bias=eos);ids=torch.tensor([[int(t.eos_token_id)]+t('Question: '+r['query']+' Answer:',add_special_tokens=True)['input_ids']]);target=torch.tensor([t(' '+r['answer_a'],add_special_tokens=False)['input_ids'][0]]);slot=b(x.unsqueeze(0));base=m.get_input_embeddings()(ids).detach().float();hidden=base.clone();hidden[:,0,:]=slot;lw=layerwise(m,hidden.detach(),2,target);b.zero_grad(set_to_none=True);slot.backward(lw['grad'][:,0,:]);analytic=torch.cat([b.weight.grad.reshape(-1),b.bias.grad.reshape(-1)]).detach();params=[b.weight.reshape(-1),b.bias.reshape(-1)];indices=[0,384,1000,10000,200000];eps=.01;numeric=[]
for idx in indices:
 pvec=params[0] if idx<params[0].numel() else params[1];j=idx if idx<params[0].numel() else idx-params[0].numel();old=float(pvec[j])
 with torch.no_grad():pvec[j]=old+eps
 lp_h=base.clone();lp_h[:,0,:]=b(x.unsqueeze(0)).detach();lp=layerwise(m,lp_h,2,target)['loss']
 with torch.no_grad():pvec[j]=old-eps
 lm_h=base.clone();lm_h[:,0,:]=b(x.unsqueeze(0)).detach();lm=layerwise(m,lm_h,2,target)['loss']
 with torch.no_grad():pvec[j]=old
 numeric.append((lp-lm)/(2*eps))
errs=[abs(float(numeric[i])-float(analytic[j]))/max(1,abs(float(numeric[i])),abs(float(analytic[j]))) for i,j in enumerate(indices)];out={'schema':'LITE-008Q-gradient-chain-v1','status':'completed','configuration':'two-layer reduced FP32 layer-wise reference','objective':'answer-token NLL','bridge':'Linear(384,576)','finite_difference_epsilon':eps,'indices':indices,'analytic_gradients':[float(analytic[i]) for i in indices],'numeric_gradients':[float(x) for x in numeric],'relative_errors':errs,'max_relative_error':max(errs),'finite_bridge_gradients':bool(torch.isfinite(analytic).all()),'bridge_output_shape':list(slot.shape),'encoder_identity':encid,'encoder_parameters_frozen':True,'decoder_parameters_frozen':all(not p.requires_grad for p in m.parameters()),'decoder_parameter_gradients_present':any(p.grad is not None for p in m.parameters()),'external_memory_positions':1,'protected_final_holdout_accessed':False};(ROOT/'gradient-chain.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'max_relative_error':max(errs),'finite':out['finite_bridge_gradients']}))
