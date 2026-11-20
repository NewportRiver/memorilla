# LITE-006 Neural Audit

## Implementation

`lite/neural` contains the compact Newport River query-conditioned associative-memory architecture:

- deterministic learned slots;
- 384-D frozen query and memory inputs;
- 128-D latent space;
- 8 fixed slots;
- 4 attention heads;
- one masked cross-attention block;
- residual LayerNorm and tanh-GELU FFN refinement;
- 384-D output slots;
- compact evidence scoring;
- safe checksummed NPZ/JSON weights.

Exact parameter count: `429,312`.
Raw Float32 weights: `1,717,248` bytes.

## Validation status

```text
NumPy forward:                 passed
NumPy full/stream agreement:   passed; <6.5e-10 max error
Serialization round-trip:      passed
Malformed weight rejection:    passed
Compatibility rejection:       passed
Scope/temporal policy gate:    passed
Empty memory behavior:         passed
Permutation invariance:        passed
PyTorch forward:               passed
PyTorch NumPy parity:          passed; 1.56e-6 max error
PyTorch nonzero gradients:     21/29 tensors
Full training:                 not run
Optimizer updates:             not run
```

PyTorch smoke RSS was `225,116,160` bytes in the final recorded run. The NumPy collection benchmark peak was `101,720,064` bytes at 5,000 records. Both remain below the approved `256 MiB` maximum observed process RSS.

## Upstream relationship

The Apache-licensed upstream `memorilla/memory.py` directly informs learned slot initialization, question conditioning, multi-head cross-attention, residual normalization/FFN refinement, fixed token count, and collection-oriented processing. The mobile implementation is not an exact Stanford reproduction. It reduces width, slots, heads, and depth; uses a NumPy streaming reference; keeps evidence policy external; omits decoder/vLLM integration; and adds non-executable provenance-checked serialization.

## Safety boundary

Neural scores and attention weights do not authorize evidence. Policy eligibility is applied before scoring, and authoritative resolution/source-hash verification is required after ranking. No language-generation bridge is implemented.

## Training boundary

Weights are random and explicitly marked untrained. LITE-007 training remains a separately authorized future experiment with Architect1 as the preferred environment subject to hardware verification.
