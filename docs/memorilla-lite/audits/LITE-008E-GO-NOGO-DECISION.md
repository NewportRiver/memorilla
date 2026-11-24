# LITE-008E GO / NO-GO Decision

## Decision: CONDITIONAL GO — interface qualification passed; hypothesis testing remains prohibited

The pinned llama.cpp runtime correctly consumed externally supplied continuous embeddings on Android ARM64 using the approved Q4_K_M SmolLM2-135M artifact.

Evidence:

- `llama_model_n_embd_inp` reported 576.
- Genuine Q4 dequantized token rows produced bit-identical final logits and greedy continuation to ordinary token-ID submission.
- A mixed sequential token/external/token sequence preserved positions, logits, and continuation.
- Zero, random, perturbed, and permuted embeddings changed logits and/or outputs reproducibly.
- 1, 2, 4, and 8 slot runs at effective contexts 256 and 512 completed below the 268,435,456-byte sampled RSS ceiling.
- Three primary repetitions were completed with exit status 0.
- Malformed inputs were rejected before native submission.
- No persistent-memory, checkpoint, retrieval, SQLite, encoder, or protected-data mutation occurred.

The decision is conditional because this milestone qualifies only the external-embedding instrument. It does not qualify Gate D, Gate E, or Gate F; it does not establish learned-memory effectiveness, semantic usefulness, projection compatibility, or whole-pipeline resource compliance. Thermal telemetry and separate KV-cache allocation remain unavailable. Greedy repeats do not establish stochastic seeded reproducibility.

Explicitly not performed: 384-to-576 projection implementation or training, associative-memory connection, fine-tuning, retrieval modification, holdout access, GPU/cloud inference, APEX/Hermes integration, and RSS-ceiling changes.

Stop boundary reached: await owner review before any projection or learned-memory work.
