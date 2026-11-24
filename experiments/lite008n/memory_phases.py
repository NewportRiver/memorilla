"""Phase-level RSS sampling for the selected layer-wise reference."""
from __future__ import annotations
import copy,json,resource,time
from pathlib import Path
import torch
from experiments.lite008n.layerwise_reference import setup,masks,layer_call,fp_components
ROOT=Path.home()/'.cache/memorilla-lite/LITE-008N'
def rss():
 try:return int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])*1024
 except:return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();tok,m,ids,target=setup();p={'after_model_load':rss()};h=m.get_input_embeddings()(ids).detach().float();p['after_embedding_fp32']=rss();posids,pos,mask,pe=masks(m,h);states=[h];peak_forward=rss()
 for i in range(30):
  layer=copy.deepcopy(m.model.layers[i]).float();peak_forward=max(peak_forward,rss())
  with torch.no_grad(): h=layer_call(layer,h,mask,posids,pos,pe).detach()
  states.append(h);peak_forward=max(peak_forward,rss());del layer
 p['after_forward']=rss();p['forward_peak_rss']=peak_forward;norm,ow=fp_components(m);h=states[-1].requires_grad_(True);logits=torch.nn.functional.linear(norm(h)[:,-1,:],ow).float();loss=torch.nn.functional.cross_entropy(logits,target);gh=torch.autograd.grad(loss,h)[0].detach();peak_backward=rss()
 for i in reversed(range(30)):
  layer=copy.deepcopy(m.model.layers[i]).float();inp=states[i].detach().requires_grad_(True);out=layer_call(layer,inp,mask,posids,pos,pe);gh=torch.autograd.grad(out,inp,grad_outputs=gh)[0].detach();peak_backward=max(peak_backward,rss());del layer,out,inp
 p['after_backward']=rss();p['backward_peak_rss']=peak_backward;out={'schema':'LITE-008N-memory-phases-v1','phases':p,'loss':float(loss),'gradient_norm':float(gh.norm()),'wall_seconds':time.monotonic()-st,'stored_weight_dtype':'bfloat16','compute_dtype':'float32','decoder_layers':30};(ROOT/'memory-phase-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
