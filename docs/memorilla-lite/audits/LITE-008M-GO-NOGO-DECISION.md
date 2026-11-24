# LITE-008M GO / NO-GO Decision

## Decision: CONDITIONAL GO for owner review; NO-GO for full-model FP32 qualification

Memory-efficient Transformers loading is compatible and removes the conventional loading-time RSS violation. It preserves tied weights and exactly matches sampled source tensors converted to Float32.

The complete FP32 reference still exceeded the unchanged 1 GiB RSS ceiling before producing a durable forward/backward result. A genuine reduced component showed excellent BF16/FP32 gradient agreement and useful negative-gradient steps, but it omits all transformer layers and therefore cannot qualify the complete decoder.

The corrected balanced textual evaluation demonstrates relevant evidence ranking (`24/24`) and counterfactual reversal (`21/24`) but only `13/24` query-only accuracy and `17/24` conflicting-evidence accuracy. This is mixed task competence, not a latent-memory authorization.

Stop at owner review. Do not resume oracle optimization, generalized bridge training, complete associative-memory training, alternative-model acquisition, or dependency installation.
