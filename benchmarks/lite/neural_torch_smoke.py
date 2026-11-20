"""Isolated LITE-006 PyTorch forward/backward smoke test; never trains or saves weights."""
import json,resource,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from lite.neural import NeuralConfig,NumpyAssociativeMemory

def main():
 import torch
 torch.set_num_threads(2); torch.set_num_interop_threads(1); resource.setrlimit(resource.RLIMIT_CPU,(30,30))
 c=NeuralConfig(); n=NumpyAssociativeMemory(c); from lite.neural.torch_model import TorchAssociativeMemory
 t=TorchAssociativeMemory(c); t.load_numpy_state(n.state); t.train()
 rng=np.random.default_rng(6); q=rng.normal(size=(1,384)).astype('float32'); mem=rng.normal(size=(1,137,384)).astype('float32'); mask=np.ones((1,137),bool); mask[:,::11]=False
 tq=torch.tensor(q,requires_grad=False); tm=torch.tensor(mem); tmask=torch.tensor(mask); out=t(tq,tm,tmask); npout=n.forward(q[0],mem[0],mask[0]); parity=float(np.max(np.abs(out.detach().numpy()[0]-npout))); loss=(out.square().mean()); loss.backward(); grads=[float(p.grad.abs().max()) if p.grad is not None else 0.0 for p in t.parameters()];
 print(json.dumps({'status':'passed','torch_version':torch.__version__,'parameter_count':sum(p.numel() for p in t.parameters()),'output_shape':list(out.shape),'finite':bool(torch.isfinite(out).all()),'numpy_max_abs_error':parity,'numpy_torch_allclose':bool(np.allclose(out.detach().numpy()[0],npout,rtol=2e-5,atol=2e-5)),'loss':float(loss.detach()),'nonzero_gradient_tensors':sum(x>0 for x in grads),'gradient_tensors':len(grads),'max_gradient':max(grads),'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},sort_keys=True))
if __name__=='__main__': main()
