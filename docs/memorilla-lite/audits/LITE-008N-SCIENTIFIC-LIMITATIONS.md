# LITE-008N Scientific Limitations

- The FP32 reference computes with official BF16-stored weights converted layer-by-layer to Float32; no separate original Float32 checkpoint was available.
- The manual layer-wise VJP is mathematically validated against a conventional two-layer FP32 composition, not against a complete simultaneously resident Float32 model.
- The complete 30-layer path includes all decoder operations and produces finite external gradients, but its exact equivalence to a simultaneously resident complete Float32 model is inferred from the reduced exact-equivalence test and implementation identity.
- Full BF16 gradient comparison is not fully qualified: cosine and norm ratio pass, coordinate sign agreement fails the retained threshold (`0.970486` versus `0.98`).
- Gradient usefulness used three bounded synthetic examples and fixed steps. It is not semantic bridge training, oracle evidence, or associative-memory training.
- Textual candidate-ranking evidence remains synthetic and preliminary.
- The first failed N qualification invocation was a harness shape error before scientific completion; no result was accepted. The corrected run is the reported result.
- No protected final holdout was accessed.
