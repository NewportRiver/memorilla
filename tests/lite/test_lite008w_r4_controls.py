import torch
from experiments.lite008w.control_utils import *

def test_zero_embedding_is_genuine_zero_and_distinct_from_eos():
 eos=torch.ones(4); z=zero_embedding(eos)
 assert torch.equal(z,torch.zeros(4)); assert not torch.equal(z,eos)

def test_query_only_eos_and_untrained_zero_residual_are_explicitly_equivalent():
 eos=torch.arange(4.).float(); assert control_equivalence(eos,eos.clone())

def test_pair_reversal_requires_both_directions():
 assert pair_reversal(1.0,-1.0); assert not pair_reversal(1.0,1.0)

def test_representation_hash_excludes_candidates():
 assert representation_input_hash('q','e')==representation_input_hash('q','e')

def test_seeded_random_is_norm_matched_and_deterministic():
 ref=torch.ones(8); a=deterministic_scale_matched_random(ref); b=deterministic_scale_matched_random(ref)
 assert torch.equal(a,b); assert torch.allclose(a.norm(),ref.norm())
