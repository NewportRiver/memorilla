# LITE-008A Experiment Design

## Feasibility status

The requested three-arm decoder experiment cannot be executed with the currently available local components. The design below is therefore a future controlled proposal, not an implementation or simulation.

## Intended arms

### Arm A — conventional retrieval context

- Query the existing SQLite/conventional retrieval path.
- Use the same evidence records that would feed the latent arm.
- Serialize the retrieved evidence as ordinary text context.
- Account for exact UTF-8 bytes, tokenizer tokens if a decoder is later available, evidence IDs, and prompt overhead.

### Arm B — compact latent memory

- Encode the same evidence with the pinned MiniLM artifact.
- Derive compact latent memory from the same evidence and query.
- Pass the result through a decoder-compatible memory-token interface.
- Account for latent tensor bytes, number of memory tokens, projection/adaptor parameters, and any serialization overhead.

The existing `lite/neural/NumpyAssociativeMemory` cannot satisfy the last step: it produces retrieval/ranking state and has no decoder interface. The broader `memorilla.MemoryModule` can produce decoder-width memory tokens, but only its PyTorch/Transformers/vLLM integration provides the required decoder path, and that path is unavailable locally on this Android environment.

### Arm C — no evidence

- Run the same query and decoder with the same prompt/template and no relevant evidence.
- Preserve all other inputs, seed, decoding policy, and output budget.
- This is the negative conditioning control.

## Smallest future compatible experiment

Only after a local CPU-compatible decoder or deliberately approved toy decoder exists:

1. Use a deterministic synthetic development task where each query requires one or two known evidence facts.
2. Create a separate sealed evaluation task with disjoint facts/templates.
3. Fix decoder identity/checksum, tokenizer, encoder identity/checksum, seed, initialization, decoding parameters, and output limit.
4. Use identical source evidence and query text in Arms A, B, and C.
5. Keep the decoder frozen initially; if a conditioning projection is required, train only that projection under a separately authorized bounded budget.
6. Predeclare the projection shape, for example `latent_dim × decoder_hidden_size` plus bias, and include its exact parameter count.
7. Measure exact context bytes/tokens for A, latent bytes/tokens and projection cost for B, and zero relevant evidence for C.
8. Run evidence omission and evidence substitution tests. The latent arm must change predictably when the supporting fact is removed or replaced.
9. Repeat each arm deterministically and run a second sealed synthetic task before making any broader claim.

## Metrics and failure behavior

Primary feasibility metrics:

- answer accuracy or exact fact recovery on the sealed task;
- bytes/tokens transferred to the decoder;
- latent representation size;
- difference between Arms B and C under evidence omission.

Mechanistic metrics:

- query/evidence conditioning sensitivity;
- output changes under evidence substitution;
- deterministic repeat agreement;
- latency and peak RSS;
- decoder failure, abstention, and malformed-output rates.

A failed or unavailable decoder must be recorded as blocked. A fake engine may validate tensor plumbing only; it cannot produce evidence for language-model conditioning.

## Resource gates

- RSS must remain below `256 MiB`.
- One numerical thread and one computational subprocess.
- CPU-only, no GPU, no cloud inference.
- No new model download, dependency, memory database, or APEX integration.
- Stop before implementation if decoder loading or prompt-embedding support exceeds the resource guard.

## Current incompatibility

The default decoder is `Qwen/Qwen3-8B`, while the Android environment has no local decoder artifact and lacks `transformers`, `vllm`, and `safetensors`. The repository's fake decoder tests are not a scientific surrogate. Consequently, no arm can currently execute a real decoder comparison without violating the directive.
