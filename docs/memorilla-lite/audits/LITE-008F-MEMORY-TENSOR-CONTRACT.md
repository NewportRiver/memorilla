# LITE-008F Memory Tensor Contract

## Existing Memorilla module

Source: `lite/neural/config.py`, `lite/neural/numpy_model.py`, and `lite/neural/torch_model.py`.

| Tensor | Shape | Status in LITE-007 ranking objective |
|---|---:|---|
| MiniLM query | `[384]` | frozen encoder output |
| MiniLM memory record | `[384]` | frozen encoder output |
| learned slots | `[8, 128]` | connected, not updated by the authorized LITE-007 projection runs |
| latent slot output before decoder bridge | `[8, 128]` | internal latent state |
| `out_w` | `[128, 384]` | disconnected from ranking objective; untrained |
| `out_b` | `[384]` | disconnected from ranking objective; untrained |
| existing memory output | `[8, 384]` | available only through the untrained output projection |
| score-slot projection | `[128, 64]` | ranking-connected |
| score-memory projection | `[384, 64]` | ranking-connected |

The complete NumPy state contains 29 tensors and 429,312 Float32 parameters. The existing ranking-gradient implementation intentionally excludes `out_w` and `out_b`; it does not provide a decoder-compatible output. Existing LITE-007 checkpoints therefore cannot be treated as decoder-aligned memory.

## Decoder bridge contract

The LITE-008E runtime reports `llama_model_n_embd_inp == 576`. The isolated LITE-008F bridge accepts one verified 384-dimensional semantic representation and emits one finite contiguous Float32 vector of shape `[576]`. Supported experimental slot counts remain 1, 2, 4, and 8. The bridge is not part of `MemoryModule`, does not alter its checkpoints, and is not connected to SQLite or retrieval.

The baseline candidate is dense linear projection plus bias:

`z = x W + b`, where `W ∈ R^(384×576)` and `b ∈ R^576`.

Parameter count is `384×576 + 576 = 221,760`. Initialization seed is `20261124`; serialization stores shape, dtype, seed, parameter hashes, and objective metadata.
