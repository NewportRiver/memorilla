from __future__ import annotations
import math
import numpy as np
from .config import NeuralConfig

def _check(x,shape,name):
 x=np.asarray(x,dtype=np.float32)
 if x.shape!=shape or not np.isfinite(x).all(): raise ValueError(f'{name} must be finite with shape {shape}')
 return x

def _gelu(x): return 0.5*x*(1.0+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x*x*x)))
def _ln(x,g,b,eps=1e-5):
 m=x.mean(axis=-1,keepdims=True); v=((x-m)**2).mean(axis=-1,keepdims=True); return ((x-m)/np.sqrt(v+eps))*g+b
def _linear(x,w,b): return x@w+b

def _softmax(logits,mask):
 if mask is not None: logits=np.where(mask[None,:],logits,-np.inf)
 if logits.shape[-1]==0 or (mask is not None and not np.any(mask)): return np.zeros_like(logits)
 mx=np.max(logits,axis=-1,keepdims=True); e=np.exp(logits-mx); e=np.where(np.isfinite(e),e,0.0); z=e.sum(axis=-1,keepdims=True)
 return e/np.maximum(z,np.finfo(np.float32).tiny)

class NumpyAssociativeMemory:
 """Small trainable-style query-conditioned slot module; weights are random/untrained in LITE-006."""
 def __init__(self,config:NeuralConfig|None=None,state=None):
  self.config=config or NeuralConfig(); self.state=self._init_state() if state is None else self._validate_state(state)
 def _init_state(self):
  c=self.config; r=np.random.default_rng(c.seed)
  def w(a,b): return (r.standard_normal((a,b),dtype=np.float32)/np.sqrt(max(a,1))).astype(np.float32)
  def z(n): return np.zeros((n,),np.float32)
  s={'slots':(r.standard_normal((c.slots,c.latent_dim),dtype=np.float32)/np.sqrt(c.latent_dim)).astype(np.float32),'query_w':w(c.input_dim,c.latent_dim),'query_b':z(c.latent_dim),'mem_k_w':w(c.input_dim,c.latent_dim),'mem_k_b':z(c.latent_dim),'mem_v_w':w(c.input_dim,c.latent_dim),'mem_v_b':z(c.latent_dim),'out_w':w(c.latent_dim,c.output_dim),'out_b':z(c.output_dim),'score_slot_w':w(c.latent_dim,c.score_dim),'score_slot_b':z(c.score_dim),'score_mem_w':w(c.input_dim,c.score_dim),'score_mem_b':z(c.score_dim)}
  for i in range(c.blocks):
   p=f'b{i}.'
   for n,a,b in [('q_w',c.latent_dim,c.latent_dim),('k_w',c.input_dim,c.latent_dim),('v_w',c.input_dim,c.latent_dim),('attn_out_w',c.latent_dim,c.latent_dim),('ff1_w',c.latent_dim,c.ffn_dim),('ff2_w',c.ffn_dim,c.latent_dim)]: s[p+n]=w(a,b)
   for n,d in [('q_b',c.latent_dim),('k_b',c.latent_dim),('v_b',c.latent_dim),('attn_out_b',c.latent_dim),('ff1_b',c.ffn_dim),('ff2_b',c.latent_dim),('ln1_g',c.latent_dim),('ln1_b',c.latent_dim),('ln2_g',c.latent_dim),('ln2_b',c.latent_dim)]: s[p+n]=np.ones((d,),np.float32) if n.endswith('_g') else z(d)
  return s
 def _validate_state(self,state):
  expected=self._init_state(); out={}
  if set(state)!=set(expected): raise ValueError('neural state parameter names mismatch')
  for k,v in expected.items():
   x=np.asarray(state[k],dtype=np.float32)
   if x.shape!=v.shape or not np.isfinite(x).all(): raise ValueError(f'invalid neural parameter {k}')
   out[k]=x.copy()
  return out
 def parameter_count(self): return sum(x.size for x in self.state.values())
 def _inputs(self,query,memory,mask):
  c=self.config; q=_check(query,(c.input_dim,),'query'); m=np.asarray(memory,dtype=np.float32)
  if m.ndim!=2 or m.shape[1]!=c.input_dim or not np.isfinite(m).all(): raise ValueError('memory must be finite [records,input_dim]')
  if mask is None: mask=np.ones((m.shape[0],),dtype=bool)
  mask=np.asarray(mask,dtype=bool)
  if mask.shape!=(m.shape[0],): raise ValueError('memory mask shape mismatch')
  return q,m,mask
 def _full_block(self,slots,memory,mask,i,return_attention=False):
  p=f'b{i}.'; s=self.state; q=_linear(_ln(slots,s[p+'ln1_g'],s[p+'ln1_b']),s[p+'q_w'],s[p+'q_b']); k=_linear(memory,s[p+'k_w'],s[p+'k_b']); v=_linear(memory,s[p+'v_w'],s[p+'v_b']); h=self.config.heads; d=self.config.latent_dim//h; q=q.reshape(self.config.slots,h,d); k=k.reshape(-1,h,d); v=v.reshape(-1,h,d); logits=self.config.attention_logit_scale*np.einsum('shd,nhd->shn',q,k)/np.sqrt(d); probs=np.transpose(_softmax(logits.reshape(self.config.slots*h,-1),mask).reshape(self.config.slots,h,-1),(0,1,2)); ctx=np.einsum('shn,nhd->shd',probs,v).reshape(self.config.slots,self.config.latent_dim); slots=slots+_linear(ctx,s[p+'attn_out_w'],s[p+'attn_out_b']); z=_ln(slots,s[p+'ln2_g'],s[p+'ln2_b']); slots=slots+_linear(_gelu(_linear(z,s[p+'ff1_w'],s[p+'ff1_b'])),s[p+'ff2_w'],s[p+'ff2_b']); return slots,probs.mean(axis=1) if return_attention else None
 def forward(self,query,memory,memory_mask=None,return_attention=False,return_latent=False):
  q,m,mask=self._inputs(query,memory,memory_mask); c=self.config; s=self.state; slots=s['slots']+_linear(q,s['query_w'],s['query_b'])[None,:]
  if m.shape[0] and np.any(mask):
   wk=_linear(m,s['mem_k_w'],s['mem_k_b']); wv=_linear(m,s['mem_v_w'],s['mem_v_b']); summary=(wk[mask]+wv[mask]).mean(axis=0); slots=slots+summary[None,:]
  att=[]
  for i in range(c.blocks):
   if np.any(mask): slots,a=self._full_block(slots,m,mask,i,return_attention); 
   else:
    p=f'b{i}.'; z=_ln(slots,s[p+'ln2_g'],s[p+'ln2_b']); slots=slots+_linear(_gelu(_linear(z,s[p+'ff1_w'],s[p+'ff1_b'])),s[p+'ff2_w'],s[p+'ff2_b']); a=np.zeros((c.slots,m.shape[0]),np.float32)
   if return_attention: att.append(a)
  out=_linear(slots,s['out_w'],s['out_b']); result=(out,att) if return_attention else out; return (result,slots) if return_latent else result
 def _stream_block(self,slots,chunks,i):
  p=f'b{i}.'; s=self.state; q=_linear(_ln(slots,s[p+'ln1_g'],s[p+'ln1_b']),s[p+'q_w'],s[p+'q_b']).reshape(self.config.slots,self.config.heads,-1); d=q.shape[-1]; mx=np.full((self.config.slots,self.config.heads),-np.inf,np.float32); seen=0
  for m,mask in chunks:
   m=np.asarray(m,np.float32); mask=np.ones((len(m),),bool) if mask is None else np.asarray(mask,bool); k=_linear(m,s[p+'k_w'],s[p+'k_b']).reshape(-1,self.config.heads,d); logits=self.config.attention_logit_scale*np.einsum('shd,nhd->shn',q,k)/np.sqrt(d); logits=np.where(mask[None,None,:],logits,-np.inf); mx=np.maximum(mx,np.max(logits,axis=-1)); seen+=int(mask.sum())
  if not seen:return slots
  den=np.zeros_like(mx); num=np.zeros((self.config.slots,self.config.heads,d),np.float32)
  for m,mask in chunks:
   m=np.asarray(m,np.float32); mask=np.ones((len(m),),bool) if mask is None else np.asarray(mask,bool); k=_linear(m,s[p+'k_w'],s[p+'k_b']).reshape(-1,self.config.heads,d); v=_linear(m,s[p+'v_w'],s[p+'v_b']).reshape(-1,self.config.heads,d); logits=self.config.attention_logit_scale*np.einsum('shd,nhd->shn',q,k)/np.sqrt(d); e=np.where(mask[None,None,:],np.exp(logits-mx[:,:,None]),0.0); den+=e.sum(axis=-1); num+=np.einsum('shn,nhd->shd',e,v)
  ctx=(num/np.maximum(den[:,:,None],np.finfo(np.float32).tiny)).reshape(self.config.slots,self.config.latent_dim); slots=slots+_linear(ctx,s[p+'attn_out_w'],s[p+'attn_out_b']); z=_ln(slots,s[p+'ln2_g'],s[p+'ln2_b']); return slots+_linear(_gelu(_linear(z,s[p+'ff1_w'],s[p+'ff1_b'])),s[p+'ff2_w'],s[p+'ff2_b'])
 def forward_stream(self,query,chunks):
  c=self.config; q=_check(query,(c.input_dim,),'query'); s=self.state; slots=s['slots']+_linear(q,s['query_w'],s['query_b'])[None,:]; chunks=list(chunks)
  if not chunks: return _linear(slots,s['out_w'],s['out_b'])
  total=None; count=0
  for m,mask in chunks:
   m=np.asarray(m,np.float32); mask=np.ones((len(m),),bool) if mask is None else np.asarray(mask,bool)
   if np.any(mask):
    wk=_linear(m,s['mem_k_w'],s['mem_k_b']); wv=_linear(m,s['mem_v_w'],s['mem_v_b']); part=(wk[mask]+wv[mask]).sum(axis=0); total=part if total is None else total+part; count+=int(mask.sum())
  if count: slots=slots+(total/count)[None,None,:]
  for i in range(c.blocks): slots=self._stream_block(slots,chunks,i)
  return _linear(slots,s['out_w'],s['out_b'])
