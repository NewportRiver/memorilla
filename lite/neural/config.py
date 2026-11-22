from __future__ import annotations
from dataclasses import dataclass,asdict
@dataclass(frozen=True)
class NeuralConfig:
 input_dim:int=384
 latent_dim:int=128
 slots:int=8
 heads:int=4
 blocks:int=1
 ffn_dim:int=256
 output_dim:int=384
 score_dim:int=64
 seed:int=20261120
 attention_logit_scale:float=1.0
 architecture_version:str='lite006-associative-v1'
 def __post_init__(self):
  if self.input_dim<=0 or self.latent_dim<=0 or self.slots<=0 or self.heads<=0 or self.blocks<=0 or self.ffn_dim<=0 or self.output_dim<=0: raise ValueError('dimensions must be positive')
  if self.latent_dim%self.heads: raise ValueError('latent_dim must divide heads')
  if not self.attention_logit_scale>0: raise ValueError('attention_logit_scale must be positive')
  if self.slots>64: raise ValueError('slot count exceeds bounded limit')
  if self.parameter_count()>500000: raise ValueError('parameter target exceeded')
 def parameter_count(self):
  # learned slots, query conditioning, memory projections, one attention block per layer,
  # output projection, FFN, norms, and compact scoring projections.
  n=self.slots*self.latent_dim
  n+=3*(self.input_dim*self.latent_dim)+3*self.latent_dim
  n+=self.latent_dim*self.output_dim+self.output_dim
  n+=self.latent_dim*self.score_dim+self.score_dim+self.input_dim*self.score_dim+self.score_dim
  d=self.latent_dim
  for _ in range(self.blocks):
   n+=(d*d+d)+(self.input_dim*d+d)+(self.input_dim*d+d)+(d*d+d)
   n+=(d*self.ffn_dim+self.ffn_dim)+(self.ffn_dim*d+d)+4*d
  return n
 def as_dict(self): return asdict(self)
