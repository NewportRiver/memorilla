# LITE-008K Numerical Corrections

The LITE-008J diagnostic used BFloat16 trainable slots and converted the loss through BFloat16 logits. LITE-008K corrected the diagnostic design without changing historical LITE-008J evidence:

- External slot masters are Float32 tensors.
- Decoder input tensors are cast to the decoder's BFloat16 dtype only at the model boundary.
- Logits are promoted to Float32 before log-softmax, NLL, and margin calculations.
- Gradients remain connected through the Float32-to-BFloat16 cast.
- Pre-update and post-update metrics are recorded separately.
- Answer tokenization is validated; no silent token substitution is used.
- Complete token sequences are retained in the oracle manifest.
- The memory slot is a dedicated position before the query and does not replace a meaningful query token.

The first optimized oracle implementation timed out after 300 seconds before durable results were written. It was not treated as a passing run or used to claim updates. The reduced-evaluation rerun completed normally and persisted all results.

Numerical requalification remained unresolved. After correcting the finite-difference implementation to compare the full contrastive objective rather than NLL alone, the best directional relative error was `0.1889`; other directions and perturbations ranged above `0.37` and up to `1.86`. This is not sufficient for a blanket numerical-gradient pass.
