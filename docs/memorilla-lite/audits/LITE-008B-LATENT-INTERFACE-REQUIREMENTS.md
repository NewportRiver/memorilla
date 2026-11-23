# LITE-008B Latent Interface Requirements

## Existing interface

The original Memorilla path is a soft-token interface, not ordinary retrieval augmentation:

```text
same evidence -> encoder embeddings -> query-conditioned MemoryModule
             -> [batch, num_memories, decoder_hidden_size] memory tokens
             -> replace <|memory|> token embeddings
             -> causal decoder inputs_embeds -> autoregressive output
```

`memorilla/memory.py:128-256` defines learned slots, query conditioning, cross-attention over document embeddings, refinement, and final projection. `MemoryConfig.output_dim` is the decoder hidden width. `memorilla/llm.py:12-37` performs exact placeholder replacement and `memorilla/llm.py:104-107` passes the resulting embeddings to the decoder.

This differs from ordinary retrieval augmentation: Arm A transfers text tokens and asks the decoder to interpret them; latent conditioning transfers learned continuous vectors occupying decoder embedding positions. The latent vectors do not themselves constitute evidence authorization or citations. Evidence identity and verification must remain outside the decoder.

## Required contracts

### Shape and dtype

- Encoder input: pinned 384-dimensional MiniLM vectors in the current Android baseline.
- Memory output: `[batch, num_memories, decoder_input_embedding_width]`.
- Candidate widths: Qwen3-0.6B `1024`, SmolLM2-135M `576`, SmolLM2-360M `960`.
- The existing `lite/neural` output is 384 and cannot be inserted directly into any listed decoder without a projection.
- The projection must be explicit, parameter-counted, initialized, tested, and separately authorized.
- Runtime dtype and conversion rules must be pinned; FP16/BF16/INT4 execution must not silently change semantic tests.

### Position and generation

- Exactly `num_memories` placeholder positions must be present and ordered deterministically.
- Soft tokens must occupy valid causal positions and participate in the decoder's attention/KV-cache path.
- Generation after the prefix must be autoregressive and repeatable under fixed sampling settings.
- A runtime must expose either externally supplied embeddings per token or an equivalent documented input-layer hook.

### Evidence sensitivity

- Removing the supporting evidence must change the latent output or decoder behavior in a predeclared direction.
- Substituting a conflicting evidence item must not produce the same output as the original evidence in every trial.
- The no-evidence arm must remain a real decoder execution, not a fixed answer or fake engine.
- Conventional text-context and latent arms must use identical source evidence, query, evaluation split, and output budget.

### Resource and fallback

- RSS must remain below the existing 256 MiB ceiling unless a new owner-approved envelope replaces it.
- One numerical thread and one computational subprocess.
- CPU-only, no cloud, no GPU, no new database, no runtime provider integration.
- If latent execution fails, deterministic conventional retrieval fallback must remain available.
- Decoder output cannot authorize tools, policy, trading, or evidence claims without the existing resolver/verifier.

## Minimum future interface qualification

Before any training or integration claim, run a frozen desktop/ARM64 contract test that verifies:

1. normal text generation;
2. exact hidden-width match;
3. one soft token changes output against a token-placeholder control;
4. repeated output identity;
5. omission and substitution sensitivity;
6. KV-cache continuation;
7. measured RSS, latency, and storage;
8. conventional fallback after injected failure.

Passing this interface test would not pass Gate D or Gate E globally. Gate E requires deterministic desktop and Android behavior; Gate D requires learned attention to materially improve association without harming evidence integrity. Both remain unpassed.
