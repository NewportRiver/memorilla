# LITE-008I Differentiation Methods

## A. Frozen PyTorch decoder

Scientifically faithful. The official Hugging Face model exposes autograd through `inputs_embeds`, while decoder parameters can be frozen. It requires loading the bfloat16 safetensors model and retaining activations for backward.

Status: source-compatible, full-model runtime blocked by the existing 256 MiB ceiling.

## B. Activation checkpointing

Could reduce saved activation memory by recomputation, but cannot remove the model-weight floor. The official safetensors artifact is already 269,060,552 bytes, above the ceiling, before Python/runtime, input gradients, logits, and checkpoint metadata. Not executed.

## C. Reduced precision

The qualified PyTorch build performed finite CPU matrix multiplication with autograd for float32, float16, and bfloat16. This is only an operator smoke test; it does not establish numerical reliability or practical speed for a full Android decoder backward pass. The official model is bfloat16 and the local configuration already reflects that dtype.

## D. Native quantized differentiation

The pinned llama.cpp source contains generic `ggml-opt` backward and AdamW/SGD infrastructure (`ggml_opt_alloc(..., backward)`, `ggml_build_backward_expand`, optimizer steps). However, the qualified LITE-008E harness and llama inference path expose forward evaluation only; no verified graph construction connects the llama decoder's externally supplied embeddings to a backward-compatible training graph. Quantized inference kernels alone are not evidence of a usable decoder training path. No native modification was attempted.

## Decision

Use PyTorch autograd as the reference mechanism if an owner-approved training-only resource envelope becomes available. Do not replace it with native quantized inference or gradient-free optimization while claiming scientific equivalence.
