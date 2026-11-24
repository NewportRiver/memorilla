# LITE-008I Training–Inference Compatibility

No compatibility claim is made between embeddings learned through an official differentiable PyTorch decoder and the qualified Q4_K_M llama.cpp decoder.

The paths differ in at least:

- PyTorch bfloat16 versus Q4_K_M quantized weights
- Hugging Face causal implementation versus llama.cpp graph implementation
- External embedding injection at `<|memory|>` positions
- Tokenizer and special-token registration
- Causal positions and left-padding conventions
- Logit arithmetic and quantization error

The existing harness establishes forward input-interface and scoring behavior for the Q4_K_M decoder. It does not establish that an embedding optimized against the official model preserves candidate ranking, likelihood margins, or counterfactual behavior under Q4_K_M.

A future compatibility study requires an approved training-only run to produce a disposable embedding artifact, then matched teacher-forced scoring on both decoders with identical token IDs, positions, and answer labels. No such artifact was produced in LITE-008I.
