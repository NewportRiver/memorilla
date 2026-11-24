# LITE-008E External-Embedding Contract

## Scope

This is an isolated Android ARM64 interface experiment. It does not connect to Memorilla retrieval, SQLite, MiniLM, neural-memory checkpoints, protected data, learned projection, training, or production inference.

## Pinned runtime

- llama.cpp: `969d7333172c522d6a60fffb8dd35288ce8df236`
- Model: `HuggingFaceTB/SmolLM2-135M`
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Quantization: Q4_K_M
- Q4 SHA-256: `39f688e705919e7cf51f505cedf283889e5ef813b1e969ac05f8461e7c83f458`
- Verified input width: `llama_model_n_embd_inp(model) == 576`
- Vocabulary: `49152`

## Actual API contract

The pinned header defines `struct llama_embd { const float * data; size_t n_rows; size_t n_embd; }`. The harness uses `llama_batch_ext_init`, then either `llama_batch_ext_add_token` or `llama_batch_ext_add_embd`, followed by `llama_batch_ext_set_pos`, `llama_batch_ext_set_output_logits`, and `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)`. Logits are read with `llama_get_logits`.

The input is one row of contiguous little-endian host `float32` values with exactly 576 values. Token and embedding entries are submitted in sequential batches because the experiment does not assume unsupported token/embedding co-submission semantics. Each batch has sequence ID 0 and an explicit causal position. Generated continuation tokens are submitted as ordinary token IDs at the next position.

The header documents that token and embedding arrays are alternatives when using the legacy batch API. The extended API additionally exposes separate add-token and add-embedding calls. `llama_batch_ext` has no public direct logits accessor; logits are obtained from the context after `llama_process`.

## Safety contract

The harness validates embedding length, finite values, external index range, and context capacity before native submission. It uses a fresh model context per process and writes only experiment output files. It does not write persistent Memorilla state.
