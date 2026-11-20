# LITE-007E Report — Bounded Dataset Loading and Native Learning

Status: passed bounded scoring-head pilot with 8 real SGD updates. LITE-007 remains incomplete.

## 1. Root cause

LITE-007D used an eager loader that parsed the entire development JSON, decoded the compressed embedding archive, built all indexes, and assembled all 960 train examples before model initialization. Individual source phases were fast: JSON metadata 15.2 ms, NPZ open 28.0 ms, record embedding decode 45.7 ms. The unbounded assembly/materialization boundary had no completion marker before the 30-second wall limit.

## 2. Optimized loading

A deterministic train-only preparation artifact was added at runtime outside Git. Preparation took 0.296 seconds. It contains indexed query and padded memory tensors, masks, identities, categories, source hashes, encoder identity, and no holdout material. Native training retrieves one example by index and filters to the same positive-and-negative usable examples as the original path.

Retrieval: 1 example 19.4 ms; 8 examples 137.7 ms. The single-example and eight-example targets passed.

## 3. Integrity and equivalence

One hundred deterministic examples were independently compared against the source path. IDs, partition/scenario membership, eligibility, labels, and exact Float32 embeddings matched. Corruption, missing artifact, wrong partition, and flushed preparation progress tests passed. Targeted loader/gradient/objective tests passed: `10 passed in 0.79s`.

## 4. Resource result

The native pilot completed in 0.470 seconds. Initialization RSS was 37.8 MiB, peak RSS was 68.6 MiB, and minimum observed available RAM was 1,412,775,936 bytes. All hard RAM, RSS, thread, subprocess, and wall-time guards passed. PyTorch and ONNX Runtime were not imported.

## 5. Learning result

- Actual optimizer updates: 8
- Trainable parameters: `score_slot_w`, `score_slot_b`, `score_mem_w`, `score_mem_b`
- Initial diagnostic loss: 0.6864654421806335
- Final diagnostic loss: 0.6860795617103577
- Change: -0.0003858804702758789
- Frozen parameters: byte-identical verification passed
- All gradient norms and parameter values: finite
- Checkpoint: written atomically and reloaded successfully

The diagnostic loss improved, but this is only controlled optimization evidence. It is not evidence of cognitive improvement or generalization.

An earlier same-session attempt was invalidated because the prepared path initially included abstention examples, creating two zero-loss updates. No weights from that attempt were published. The corrected pilot excluded those examples and produced the reported result.

## 6. Independent development evaluation

Eight separate eligible development examples, not used in the eight-update sequence, were evaluated after training. Mean ranking loss was 0.6915164664387703. This is an evaluation observation only; there is no baseline comparison and no claim of generalization improvement.

## 7. Checkpoint

The checkpoint is a non-executable tensor archive plus metadata under `~/.cache/memorilla-lite/LITE-007E-NUMPY-PILOT-FINAL/`. Reloaded models produced identical inference outputs. Generated weights remain outside Git.

## 8. Regression and remaining blockers

Targeted LITE tests passed. The complete inherited LITE regression remains incomplete because the upstream fixture imports unavailable `transformers`; no installation or silent exclusion was performed.

No memory-projection updates were performed. Full associative-memory training, checkpoint selection, Android trained-checkpoint inference acceptance, holdout evaluation, and production integration remain incomplete. LITE-007 is not complete.
