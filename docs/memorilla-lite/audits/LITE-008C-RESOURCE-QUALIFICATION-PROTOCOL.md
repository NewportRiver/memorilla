# LITE-008C Resource Qualification Protocol

## Status

Protocol design only. Do not download, build, load, or execute a decoder under this directive.

Reference environment from LITE-008A:

- Physical RAM: `5,660,758,016` bytes
- Available RAM: `1,622,192,128` bytes
- Free storage: `7,481,962,496` bytes
- RSS ceiling: `268,435,456` bytes
- Numerical threads: `1`
- Computational subprocesses: `1`
- CPU-only, no GPU, no cloud

## Phase 0: preflight

1. Verify branch, baseline, and clean tree.
2. Verify MiniLM checksum, SQLite snapshot identity, neural checkpoint hashes, and audit provenance.
3. Verify no holdout paths are mounted or selected.
4. Verify available RAM and storage.
5. Verify isolated model/build paths and capture their sizes before execution.
6. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and the llama.cpp runtime thread count to `1`.
7. Abort if available RAM falls below the existing guard or if free storage cannot hold the pinned source, model, temporary conversion output, and logs.

## Phase 1: ordinary text generation only

This is the first authorized decoder experiment. No latent memory and no projection are allowed.

Use a fixed short prompt and fixed greedy decoding. Record:

- model-load start/end and latency;
- process RSS immediately before/after load;
- `ru_maxrss` and `/proc/<pid>/status` VmRSS;
- prefill latency for a fixed prompt;
- generation latency and generated token count;
- tokens/second;
- context length and KV-cache configuration;
- storage before/after model load;
- CPU utilization and thread count;
- thermal state before/during/after where readable;
- process exit status, OOM, termination, or crash evidence.

The run fails if RSS exceeds `268,435,456` bytes, the process is killed, output is non-deterministic under the fixed seed/greedy settings, or the decoder cannot complete ordinary text generation.

Do not use swap to claim compliance. Swap may be recorded as system state but cannot relax the RSS gate.

## Phase 2: repeatability and budget sweep

Only if Phase 1 passes:

- repeat the same prompt three times;
- test predeclared context lengths, for example 128, 256, and 512 tokens;
- measure p50/p95 load, prefill, generation, and tokens/second;
- record KV-cache memory deltas by context length;
- stop at the first RSS, process-death, thermal, or storage failure.

No tuning against retrieval quality occurs in this phase.

## Qualification outputs

Produce an immutable JSON manifest containing:

- decoder/model revision and artifact SHA-256;
- llama.cpp source commit and binary/compiler hashes;
- command lines and environment variables;
- prompt/tokenization identity;
- all measurements and unavailable fields;
- pass/fail status for each gate;
- failure logs and cleanup record.

## Stop conditions

Stop immediately on:

- RSS above 256 MiB;
- OOM or process termination;
- decoder output failure;
- non-deterministic repeated output;
- thermal or battery guard failure;
- storage shortfall;
- any need for GPU/cloud/new dependency;
- any mutation outside the isolated LITE-008C paths.

Ordinary decoder execution does not pass Gate D, Gate E, or Gate F. It is only a prerequisite resource qualification.
