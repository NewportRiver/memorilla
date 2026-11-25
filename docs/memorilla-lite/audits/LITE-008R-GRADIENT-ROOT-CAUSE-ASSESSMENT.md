# LITE-008R gradient root-cause assessment

Stage A synthetic closed-form/autograd/finite-difference bridge passed. Stage B genuine MiniLM feature to linear bridge VJP passed with zero measured error. Therefore parameter flattening, bridge shape, and basic restoration are not the primary cause.

The reduced external-slot decoder path failed Stage C: the direct conventional decoder gradient and retained manual layer-wise VJP disagree. The ordinary-token reduced equivalence already retained in LITE-008N passes, so the evidence localizes the issue to the manual VJP when applied to the bridge-injected hidden sequence, not to generic bridge propagation.

The exact single-line cause remains unresolved. Candidate areas are the external-slot hidden construction/attention-position path and the layer-wise reverse VJP contract. The retained Q harness also did not compare the exact manual VJP against conventional autograd before training. This uncertainty is explicitly unresolved; no training result is promoted.

The Q five-coordinate discrepancy is reproduced in the preserved Q artifact and R full-chain diagnostic.
