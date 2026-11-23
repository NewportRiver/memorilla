# LITE-008C Soft-Token Contract

## Current status

No projection, decoder integration, or soft-token injection is implemented. This document defines a future contract only.

## Width and projection

- Current compact evidence representation: 384 dimensions.
- SmolLM2-135M decoder input width: 576 dimensions.
- Required future projection: `P: R^384 -> R^576`, with an explicit bias only if scientifically justified.
- The projection's parameter count, initialization, dtype, checksum, and training status must be recorded.
- A `MemoryModule(output_dim=576)` is an alternative architecture configuration, but changing the existing 384-dimensional module is an implementation decision requiring separate authorization.
- No implicit padding, truncation, broadcasting, or dtype cast may be used to hide the width mismatch.

## Token placement

- Select a fixed memory-token count before execution; preserve it across Arms A, B, and C.
- Insert exactly that many placeholders at fixed causal positions in a prompt template.
- Verify the tokenizer maps each placeholder to one intended position, not multiple subword tokens.
- Record input IDs, positions, attention mask, and sequence length.
- Preserve ordinary question/system tokens and generation prompt identically across arms.

## Embedding and numerical contract

- Compute memory vectors from the same evidence and query used by conventional retrieval.
- Convert only at an explicitly recorded boundary to the runtime/model dtype.
- Compare pre-conversion FP32 vectors with runtime vectors using absolute/relative error.
- Verify finite values, row count, width, and memory-token ordering.
- Do not claim semantic equivalence between a text token and a soft token; test output behavior directly.

## Causal and KV-cache behavior

Before any learned experiment, verify with a real decoder:

1. normal text-only generation;
2. placeholder prompt with ordinary token embeddings;
3. the same prompt with externally supplied 576-dimensional embeddings;
4. logits/output change at the expected positions;
5. autoregressive continuation after the prefix;
6. KV-cache continuation produces the same result as uncached generation within a declared tolerance;
7. repeated greedy runs are byte-identical or numerically within the declared tolerance.

For llama.cpp, qualify `llama_batch.embd` or `llama_batch_ext_add_embd` at the pinned source commit and target build. Source declarations alone are insufficient.

## Evidence tests

Use a sealed synthetic task with two known facts:

- omission: remove the supporting record and require the latent output to change or abstain;
- substitution: replace it with a conflicting record and require a distinct result;
- no evidence: no relevant record reaches the memory module;
- conventional fallback: if latent execution fails, return verified retrieved evidence rather than an unsupported answer.

Track exact source IDs outside the latent vector. Latent output cannot authorize policy, tools, trading, or provenance claims.

## Gate boundaries

- Interface pass is not Gate D.
- Interface pass is not Gate E until both desktop and Android executions are deterministic.
- Interface pass is not Gate F until resource, thermal, process-death, and fallback tests pass.
- No training or learned-memory effectiveness claim follows from shape or plumbing tests alone.
