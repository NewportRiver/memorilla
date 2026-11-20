# LITE-007C Report — Native Android Learning Engine

Status: implementation and mathematical verification passed; controlled pilot blocked before training by the RAM guard.

## Implemented

- `lite/training/numpy_gradients.py`: exact scoring-head gradients for normalized slot/evidence projections and multi-positive logistic ranking.
- `lite/training/numpy_optimizer.py`: plain in-place Float32 SGD, clipping support, and frozen-parameter verification.
- `lite/training/numpy_telemetry.py`: flushed phase markers with RSS and available-memory measurements.
- `lite/training/numpy_train.py`: bounded, authorization-gated pilot with one thread, one subprocess, RAM/RSS/time guards, deterministic ordering, atomic run metadata, and existing non-executable weight serialization.
- NumPy model latent return support for training the existing architecture without changing its structure.
- Lazy training-package imports so the NumPy subprocess does not initialize PyTorch.
- Atomic metadata writes for the existing weight serialization format.

Only the existing scoring head is trainable: `score_slot_w`, `score_slot_b`, `score_mem_w`, and `score_mem_b`. All other architecture parameters remain frozen.

## Mathematical verification

- Finite difference: passed.
  - `score_slot_w[0,0]`
  - epsilon `0.001`
  - absolute error `8.93e-06`
- PyTorch reference: passed in a separate process.
  - loss absolute error `0`
  - maximum gradient absolute error `5.59e-09`
- NumPy gradient tests: `4 passed in 0.19s`
- Covered multi-positive ranking, hard negatives, empty positives, nonfinite inputs, determinism, clipping, and frozen parameters.
- Checkpoint round-trip: passed.
- Corruption rejection: passed.

During reference comparison, an existing sign error in the shared PyTorch pairwise objective was detected and corrected from `positive - negative` to `negative - positive`, matching the documented ranking objective. NumPy and PyTorch now agree.

## Pilot result

The explicitly bounded pilot was invoked once with the requested limits. It stopped before dataset loading and model initialization:

- Required minimum RAM: `1,342,177,280` bytes
- Observed available RAM: `1,333,682,176` bytes
- Initial process RSS: `219,283,456` bytes
- Actual optimizer updates: `0`
- Initial loss: not measured
- Final loss: not measured
- Checkpoint: none
- Automatic retry: no

This is a safe resource block, not a training failure and not evidence of learning or generalization.

## Checkpoint behavior

The implementation reuses the existing non-executable `.npz` plus JSON metadata format. Metadata writes are now atomic. The pilot writes outside Git and would include architecture, encoder, dataset, training status, and parameter checksums if it reached checkpoint serialization.

## Next development stage

After a successful authorized scoring-head pilot, the next bounded stage should add selected `mem_k_*` and `mem_v_*` projections. It requires gradients through both masked summary initialization and attention key/value paths, plus new finite-difference and PyTorch parity tests. Query-conditioning and slot initialization should follow only after that stage is independently verified. Full manual backpropagation through the complete architecture is not recommended on Android under the current RSS ceiling.

LITE-007 remains incomplete. No full associative-memory training, holdout evaluation, or production integration was performed.
