# LITE-007D Report — First Native Learning Execution

Status: objective and gradient verification passed; the first Android pilot did not reach training because dataset loading exceeded the 30-second subprocess limit. No optimizer update occurred.

## Objective audit

The corrected objective is consistently:

`softplus(margin + negative_score - positive_score)`

Directional tests passed for positive and negative score changes, a small SGD descent step, and four representative scoring-head finite differences. The isolated PyTorch parity result remains a maximum gradient absolute error of `5.59e-09`.

Earlier LITE-007 preparation occurred before the sign correction. It contained no optimizer updates or trained checkpoints. Structural gradient/connectivity observations remain valid; any hypothetical optimization interpretation under the old sign is invalidated.

## Resource investigation

Fresh-process phase measurements:

- Python start: `12.4 MiB` RSS in a clean measurement process
- NumPy import: `25.0 MiB` RSS
- `numpy_train` module import: `29.1 MiB` RSS
- PyTorch imported: no
- ONNX Runtime imported: no

LITE-007D pilot telemetry:

- Python/process start: `16.3 MiB` RSS
- After NumPy import: `29.5 MiB` RSS
- Available RAM at process start: `1,534,373,888` bytes
- Available RAM after NumPy import: `1,524,322,304` bytes
- Dataset-loading phase started at approximately `0.132` seconds
- Dataset-loading completion marker was never written
- No model, gradient, optimizer, or checkpoint phase was reached

The prior `219,283,456`-byte initial RSS was not reproduced in a clean native process. It was likely inherited/process-context overhead, but it is not subtracted from any measurement or reclassified as absent.

## Pilot result

- Attempt count: 1
- Maximum updates: 8
- Actual updates: 0
- Initial loss: not measured
- Final loss: not measured
- Checkpoint: none
- Automatic retry: no
- Orphaned process after termination: no

The pilot was terminated by the hard 30-second wall limit during dataset loading. This is a blocked execution, not a learning result.

## Regression

- Objective and gradient tests: `7 passed in 0.27s`
- Fresh native import isolation: passed; no PyTorch or ONNX imports
- Full inherited LITE regression remains blocked by the upstream Transformers fixture dependency.
- Prior isolated suite timeout remains recorded as unexecuted rather than passed.

## Checkpoint

No LITE-007D checkpoint was generated. Existing serialization round-trip and corruption rejection had passed before this attempt; no checkpoint reload was possible in this attempt.

## Next stage

The next development authorization should optimize the bounded dataset-loading path without relaxing any resource guard. It must preserve precomputed embeddings, avoid holdout access, and add a dataset-loading completion marker before another pilot is considered. No memory-projection updates are authorized by this mission.

LITE-007 remains incomplete.
