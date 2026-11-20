# LITE-006 — Trainable Query-Conditioned Associative Memory Architecture

## Status

Completed for bounded architecture implementation and mathematical validation. No full neural training, optimizer update, checkpoint selection, holdout evaluation, or learned cognitive claim occurred.

## Deliverables

- Compact NumPy reference with deterministic learned initialization.
- Lazy PyTorch reference with forward/backward smoke validation.
- Fixed query-conditioned 8-slot output independent of collection size.
- Masked full and streamed global-softmax cross-attention.
- Compact slot-to-evidence scoring interface.
- Multi-positive ranking and diversity objective preparation.
- Evidence-policy gate outside the neural model.
- Non-executable versioned checksummed weight serialization.
- LITE-005 correction audit and corrected benchmark artifacts.
- LITE-006 mathematical tests and bounded collection benchmark.
- LITE-007 training protocol preparation.

## Configuration and measured size

```text
384 input / 128 latent / 8 slots / 4 heads / 1 block / 256 FFN / 384 output
429,312 parameters
1,717,248 raw Float32 weight bytes
```

## Validation

- NumPy architecture tests: passed.
- Full versus streamed attention: maximum error below 6.5e-10.
- PyTorch forward/backward subprocess: passed.
- NumPy/PyTorch parity: maximum error 1.56e-6.
- Nonzero gradients: 21/29 tensors.
- PyTorch smoke RSS: 225,116,160 bytes in the final recorded run.
- NumPy 5,000-record benchmark RSS: 101,720,064 bytes in the final rerun.
- Existing LITE-002 through LITE-005 regressions: required before release.

## Scientific boundary

The module is randomly initialized and untrained. It demonstrates an architecture and bounded numerical behavior, not learned memory effectiveness. Existing conventional retrieval remains the control. Evidence verification remains authoritative and external to the model.

## Upstream relationship

The design preserves upstream Memorilla principles from the Apache-licensed `memorilla/memory.py`: learned slots, query conditioning, multi-head cross-attention, residual/normalization/FFN refinement, and fixed output token count. Newport River adaptations are the reduced 8-slot/128-latent Android profile, NumPy streaming reference, external evidence-policy gate, safe tensor serialization, and removal of decoder/training integration.

## Next milestone

Stop after LITE-006 and await owner authorization for LITE-007 training. Architect1 is the preferred future training environment subject to hardware verification.
