from __future__ import annotations
try:
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
except ImportError: torch=None; nn=object; F=None
from .config import NeuralConfig

if torch is not None:
 class _Block(nn.Module):
  def __init__(self,c):
   super().__init__(); d=c.latent_dim; self.q_w=nn.Parameter(torch.empty(d,d)); self.q_b=nn.Parameter(torch.zeros(d)); self.k_w=nn.Parameter(torch.empty(c.input_dim,d)); self.k_b=nn.Parameter(torch.zeros(d)); self.v_w=nn.Parameter(torch.empty(c.input_dim,d)); self.v_b=nn.Parameter(torch.zeros(d)); self.attn_out_w=nn.Parameter(torch.empty(d,d)); self.attn_out_b=nn.Parameter(torch.zeros(d)); self.ff1_w=nn.Parameter(torch.empty(d,c.ffn_dim)); self.ff1_b=nn.Parameter(torch.zeros(c.ffn_dim)); self.ff2_w=nn.Parameter(torch.empty(c.ffn_dim,d)); self.ff2_b=nn.Parameter(torch.zeros(d)); self.ln1_g=nn.Parameter(torch.ones(d)); self.ln1_b=nn.Parameter(torch.zeros(d)); self.ln2_g=nn.Parameter(torch.ones(d)); self.ln2_b=nn.Parameter(torch.zeros(d))
  def reset(self,g):
   for p in (self.q_w,self.k_w,self.v_w,self.attn_out_w,self.ff1_w,self.ff2_w): nn.init.normal_(p,0,1/max(p.shape[0],1)**0.5)
 class TorchAssociativeMemory(nn.Module):
  def __init__(self,config:NeuralConfig|None=None):
   super().__init__(); self.config=config or NeuralConfig(); c=self.config; g=torch.Generator().manual_seed(c.seed); self.slots=nn.Parameter(torch.randn(c.slots,c.latent_dim,generator=g)/c.latent_dim**0.5); self.query_w=nn.Parameter(torch.randn(c.input_dim,c.latent_dim,generator=g)/c.input_dim**0.5); self.query_b=nn.Parameter(torch.zeros(c.latent_dim)); self.mem_k_w=nn.Parameter(torch.randn(c.input_dim,c.latent_dim,generator=g)/c.input_dim**0.5); self.mem_k_b=nn.Parameter(torch.zeros(c.latent_dim)); self.mem_v_w=nn.Parameter(torch.randn(c.input_dim,c.latent_dim,generator=g)/c.input_dim**0.5); self.mem_v_b=nn.Parameter(torch.zeros(c.latent_dim)); self.out_w=nn.Parameter(torch.randn(c.latent_dim,c.output_dim,generator=g)/c.latent_dim**0.5); self.out_b=nn.Parameter(torch.zeros(c.output_dim)); self.score_slot_w=nn.Parameter(torch.randn(c.latent_dim,c.score_dim,generator=g)/c.latent_dim**0.5); self.score_slot_b=nn.Parameter(torch.zeros(c.score_dim)); self.score_mem_w=nn.Parameter(torch.randn(c.input_dim,c.score_dim,generator=g)/c.input_dim**0.5); self.score_mem_b=nn.Parameter(torch.zeros(c.score_dim)); self.blocks=nn.ModuleList([_Block(c) for _ in range(c.blocks)])
   for b in self.blocks:b.reset(g)
  def forward(self,query,memory,memory_mask=None,return_attention=False):
   c=self.config
   if query.ndim!=2 or query.shape[1]!=c.input_dim or memory.ndim!=3 or memory.shape[0]!=query.shape[0] or memory.shape[2]!=c.input_dim: raise ValueError('invalid query/memory shapes')
   if not torch.isfinite(query).all() or not torch.isfinite(memory).all(): raise ValueError('non-finite neural input')
   B,N,_=memory.shape; mask=torch.ones((B,N),dtype=torch.bool,device=memory.device) if memory_mask is None else memory_mask.to(torch.bool)
   if mask.shape!=(B,N): raise ValueError('memory mask shape mismatch')
   slots=self.slots.unsqueeze(0).expand(B,-1,-1)+F.linear(query,self.query_w.T,self.query_b).unsqueeze(1); attns=[]; h=c.heads; d=c.latent_dim//h
   for b in self.blocks:
    q=F.linear(F.layer_norm(slots,(c.latent_dim,),b.ln1_g,b.ln1_b),b.q_w.T,b.q_b).reshape(B,c.slots,h,d); k=F.linear(memory,b.k_w.T,b.k_b).reshape(B,N,h,d); v=F.linear(memory,b.v_w.T,b.v_b).reshape(B,N,h,d)
    if N==0 or not bool(mask.any()): context=torch.zeros_like(slots); probs=torch.zeros((B,h,c.slots,N),device=memory.device,dtype=memory.dtype)
    else:
     logits=torch.einsum('bshd,bnhd->bhsn',q,k)/d**0.5; logits=logits.masked_fill(~mask[:,None,None,:],-torch.inf); probs=torch.softmax(logits,dim=-1); context=torch.einsum('bhsn,bnhd->bshd',probs,v).reshape(B,c.slots,c.latent_dim)
    slots=slots+F.linear(context,b.attn_out_w.T,b.attn_out_b); z=F.layer_norm(slots,(c.latent_dim,),b.ln2_g,b.ln2_b); slots=slots+F.linear(F.gelu(F.linear(z,b.ff1_w.T,b.ff1_b),approximate='tanh'),b.ff2_w.T,b.ff2_b); attns.append(probs)
   out=F.linear(slots,self.out_w.T,self.out_b); return (out,attns) if return_attention else out
  def load_numpy_state(self,state):
   with torch.no_grad():
    for name,x in state.items():
     target={'slots':self.slots,'query_w':self.query_w,'query_b':self.query_b,'mem_k_w':self.mem_k_w,'mem_k_b':self.mem_k_b,'mem_v_w':self.mem_v_w,'mem_v_b':self.mem_v_b,'out_w':self.out_w,'out_b':self.out_b,'score_slot_w':self.score_slot_w,'score_slot_b':self.score_slot_b,'score_mem_w':self.score_mem_w,'score_mem_b':self.score_mem_b}.get(name)
     if target is None:
      i=int(name.split('.')[0][1:]); target=getattr(self.blocks[i],name.split('.',1)[1])
     if tuple(target.shape)!=tuple(x.shape): raise ValueError(f'shape mismatch: {name}')
     target.copy_(torch.from_numpy(x))
   return self
  def export_numpy_state(self):
   out={k:v.detach().cpu().numpy().astype('float32').copy() for k,v in {'slots':self.slots,'query_w':self.query_w,'query_b':self.query_b,'mem_k_w':self.mem_k_w,'mem_k_b':self.mem_k_b,'mem_v_w':self.mem_v_w,'mem_v_b':self.mem_v_b,'out_w':self.out_w,'out_b':self.out_b,'score_slot_w':self.score_slot_w,'score_slot_b':self.score_slot_b,'score_mem_w':self.score_mem_w,'score_mem_b':self.score_mem_b}.items()}
   for i,b in enumerate(self.blocks):
    for n in ('q_w','q_b','k_w','k_b','v_w','v_b','attn_out_w','attn_out_b','ff1_w','ff1_b','ff2_w','ff2_b','ln1_g','ln1_b','ln2_g','ln2_b'):out[f'b{i}.{n}']=getattr(b,n).detach().cpu().numpy().astype('float32').copy()
   return out
else:
 class TorchAssociativeMemory:
  def __init__(self,*a,**k): raise RuntimeError('PyTorch is unavailable')
