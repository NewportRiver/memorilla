import hashlib,json,torch

def zero_embedding(eos):
    return torch.zeros_like(eos)

def query_only_eos_slot(eos):
    return eos.clone()

def pair_reversal(margin_a, margin_b):
    return bool(margin_a > 0 and margin_b < 0)

def representation_input_hash(query, evidence):
    return hashlib.sha256((query+'\0'+evidence).encode()).hexdigest()

def deterministic_scale_matched_random(reference, seed=8008):
    g=torch.Generator(device='cpu').manual_seed(seed)
    x=torch.randn(reference.shape,generator=g,dtype=reference.dtype)
    return x/x.norm()*reference.norm()

def control_equivalence(eos, untrained_projection):
    return bool(torch.equal(query_only_eos_slot(eos),untrained_projection))
