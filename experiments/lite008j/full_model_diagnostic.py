"""LITE-008J full-model frozen-decoder diagnostic.

Runs inside one monitored subprocess with the local qualified HF environment.
No decoder parameter updates or checkpoint writes occur.
"""
from __future__ import annotations
import json, os, resource, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path.home()/'.cache/memorilla-lite/LITE-008J'; ROOT.mkdir(parents=True,exist_ok=True)
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
SEED=20261124

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024

def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(SEED)
    start=time.monotonic(); load_start=start
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    load_seconds=time.monotonic()-load_start
    model.requires_grad_(False); model.eval(); model.config.use_cache=False
    for p in model.parameters(): assert not p.requires_grad
    embedding=model.get_input_embeddings()
    prompt='The code for entity cobalt is'
    ids=tokenizer(prompt,return_tensors='pt',add_special_tokens=True)['input_ids']
    target_ids=tokenizer(' red',add_special_tokens=False)['input_ids']; alt_ids=tokenizer(' blue',add_special_tokens=False)['input_ids']
    if len(target_ids)!=1 or len(alt_ids)!=1:
        target_ids=[tokenizer.convert_tokens_to_ids('red')]; alt_ids=[tokenizer.convert_tokens_to_ids('blue')]
    target=int(target_ids[0]); alternate=int(alt_ids[0]); mask=torch.ones_like(ids)
    assert ids.shape[1] <= 32 and ids.shape[1] >= 3
    with torch.no_grad():
        token_embeds=embedding(ids)
        token_logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
        external_logits=model(inputs_embeds=token_embeds,attention_mask=mask,use_cache=False).logits
    max_diff=float((token_logits-external_logits).float().abs().max())
    forward={'sequence_length':int(ids.shape[1]),'hidden_size':int(token_embeds.shape[-1]),'max_abs_logit_difference':max_diff,
             'tolerance':0.02,'equivalent':max_diff<=0.02,'forward_seconds':time.monotonic()-start}
    assert forward['equivalent']
    base_slot=token_embeds[:,1,:].detach().clone()
    def make_emb(slot):
        out=token_embeds.detach().clone();out[:,1,:]=slot;return out
    def loss_for(slot, answer):
        logits=model(inputs_embeds=make_emb(slot),attention_mask=mask,use_cache=False).logits[:,-1,:]
        return torch.nn.functional.cross_entropy(logits,torch.tensor([answer]))
    slot=base_slot.detach().clone().requires_grad_(True)
    backward_start=time.monotonic(); loss=loss_for(slot,target); loss.backward(); grad=slot.grad.detach().clone()
    backward={'loss':float(loss.detach()),'gradient_shape':list(grad.shape),'gradient_dtype':str(grad.dtype),
              'gradient_norm':float(grad.float().norm()),'gradient_finite':bool(torch.isfinite(grad).all()),
              'gradient_nonzero':bool(grad.float().norm()>0),'backward_seconds':time.monotonic()-backward_start,
              'decoder_parameters_frozen':all(not p.requires_grad for p in model.parameters()),
              'decoder_parameter_grads_absent':all(p.grad is None for p in model.parameters())}
    assert backward['gradient_finite'] and backward['gradient_nonzero'] and backward['decoder_parameter_grads_absent']
    # BF16 directional finite difference; use a large perturbation appropriate to BF16 input arithmetic.
    direction=torch.randn_like(slot); direction/=direction.float().norm()
    eps=0.5
    with torch.no_grad():
        fd_plus=float(loss_for(slot.detach()+eps*direction,target));fd_minus=float(loss_for(slot.detach()-eps*direction,target))
    analytic=float((grad.float()*direction.float()).sum()); numeric=(fd_plus-fd_minus)/(2*eps)
    fd={'epsilon':eps,'analytic_directional_gradient':analytic,'numerical_directional_gradient':numeric,
        'relative_error':abs(analytic-numeric)/max(abs(analytic),1e-4),'plus_loss':fd_plus,'minus_loss':fd_minus,
        'precision_note':'BF16 directional check uses eps=0.5; coordinate-scale perturbations are quantized.'}
    # Embedding-only optimization, max 32 updates.
    torch.manual_seed(SEED+1); random_slot=torch.randn_like(base_slot); zero_slot=torch.zeros_like(base_slot)
    def metrics(slot_value):
        with torch.no_grad():
            lc=float(loss_for(slot_value,target)); la=float(loss_for(slot_value,alternate))
        return {'correct_loss':lc,'alternate_loss':la,'correct_minus_alternate_logprob':la-lc}
    initial=metrics(base_slot); zero=metrics(zero_slot); random=metrics(random_slot)
    optimized=base_slot.detach().clone().requires_grad_(True); opt=torch.optim.SGD([optimized],lr=0.2)
    trajectory=[]; optimize_start=time.monotonic()
    for step in range(32):
        opt.zero_grad(); current=loss_for(optimized,target); current.backward(); opt.step()
        trajectory.append({'step':step+1,'loss':float(current.detach()),'gradient_norm':float(optimized.grad.float().norm())})
    optimized_metrics=metrics(optimized.detach())
    np.save(ROOT/'optimized_embedding.npy',optimized.detach().float().numpy()); np.asarray(ids[0],dtype='<i4').tofile(ROOT/'input_ids.bin'); np.asarray([target],dtype='<i4').tofile(ROOT/'target_ids.bin')
    result={'schema':'LITE-008J-full-model-diagnostic-v1','model_path':str(MODEL),'model_dtype':str(next(model.parameters()).dtype),
      'parameter_count':sum(p.numel() for p in model.parameters()),'threads':torch.get_num_threads(),'sequence_length':int(ids.shape[1]),'external_slot_index':1,
      'target_token_id':target,'alternate_token_id':alternate,'loading':{'seconds':load_seconds,'rss_ru_max_bytes':rss()},'forward':forward,'backward':backward,
      'gradient_verification':fd,'optimization':{'updates':32,'learning_rate':0.2,'initial':initial,'zero':zero,'random':random,'optimized':optimized_metrics,'trajectory':trajectory,'seconds':time.monotonic()-optimize_start},
      'decoder_weights_updated':False,'protected_final_holdout_accessed':False,'total_seconds':time.monotonic()-start}
    (ROOT/'full-model-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__': main()
