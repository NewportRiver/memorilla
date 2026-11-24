# LITE-008E Experiment Protocol

## Predeclared controls

- Decoder: Q4_K_M SmolLM2-135M, CPU-only, `n_gpu_layers=0`
- Threads: one generation and one batch thread
- Contexts: 256 and 512; no separate 128 qualification
- Prompt: `The capital of France is`
- Token IDs: `[504, 3575, 282, 4649, 314]`
- Input width: 576
- Input dtype: contiguous little-endian `float32`
- Sequence ID: 0
- Positions: explicit 0 through N-1
- Continuation: greedy argmax, bounded at four tokens
- Equivalence tolerance: exact float32 equality when the external rows are dequantized rows from the same Q4 GGUF; any nonzero difference is reported. This is stricter than a model-quantization tolerance because it tests the runtime path itself.
- Sensitivity: fixed RNG seed `20261124`, normal random scale `0.02`, perturbation scale `0.01`
- RSS ceiling: 268,435,456 bytes
- Timeout: 120 seconds per subprocess

## Experiments

A compares ordinary token IDs against genuine Q4 input rows extracted by dequantizing `token_embd.weight` from the already-qualified Q4 GGUF. B replaces exactly token index 1 with that genuine external row and compares continuation against all-token input. C evaluates genuine, zero, fixed-seed random, perturbed, and reversed-row inputs. D evaluates 1, 2, 4, and 8 external slots at contexts 256 and 512.

Each accepted primary path was repeated three times. Raw inputs, per-run metadata, logits, and telemetry remain under `$HOME/.cache/memorilla-lite/LITE-008E/` and are intentionally not committed.

The previously observed seed-reporting discrepancy is not used as evidence of stochastic reproducibility: all primary runs are greedy. The harness records the fixed experiment seed only for synthetic input construction.
