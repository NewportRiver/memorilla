# LITE-008I GO / NO-GO Decision

## Decision: CONDITIONAL GO for the original differentiable mechanism; NO-GO for full-model Android execution under the current ceiling

Positive evidence:

- Upstream source inspection confirms a genuine differentiable path from answer-only causal loss through injected memory embeddings into MemoryModule parameters.
- Decoder freezing is explicit and optimizer selection is memory-only.
- Label construction masks left padding and prompt positions with `-100` and supervises answer tokens only.
- The isolated synthetic frozen-decoder test produced finite external-embedding gradients, no decoder parameter gradients, finite-difference relative error `6.65e-5`, and loss reduction from `1.4727` to `0.1125`.
- Local PyTorch autograd smoke tests worked for float32, float16, and bfloat16 CPU tensors.
- Native GGML contains generic backward/optimizer infrastructure, but no verified llama external-embedding training path.

Blocking evidence:

- Official bfloat16 safetensors artifact: `269,060,552` bytes.
- Existing decoder ceiling: `268,435,456` bytes.
- This weight floor exceeds the ceiling before activations, gradients, logits, allocator overhead, and Python runtime.
- Full-model forward/backward was therefore not executed and no peak RSS was fabricated.

Unexecuted:

- Full official SmolLM2 frozen-decoder backward pass
- Activation-checkpointed full-model pass
- Full-model float16/bfloat16 comparison
- Native quantized differentiation
- Training-to-Q4 compatibility
- Any general bridge or associative-memory training

Gate D, Gate E, and Gate F remain unpassed. Owner review is required before any training-only resource envelope larger than the current inference ceiling is used.
