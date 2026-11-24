# LITE-008F Learning Methods Analysis

## Method A — differentiable frozen decoder

Not selected. The pinned llama.cpp external-embedding API is an inference interface and exposes no backward graph or input-embedding gradient. Frozen weights would still require storing or recomputing decoder activations for reverse mode. The Q4 runtime is not a training/autograd runtime. Introducing a second inference/training engine is outside this milestone.

## Method B — decoder embedding-space alignment

Selected as the smallest defensible feasibility method. A lightweight projection is trained against genuine decoder input embeddings extracted from the already-qualified Q4 GGUF. This has a reproducible MSE objective, finite-difference verification, deterministic serialization, and a small parameter count. It measures reconstruction/alignment only; downstream decoder answer behavior must be evaluated independently.

## Method C — gradient-free decoder feedback

Not selected. The available harness can observe logits, but a useful likelihood objective would require repeated decoder subprocess evaluations, a high-dimensional 221,760-parameter search, and a large evaluation budget. No evidence supports practical or scientifically interpretable optimization under the one-subprocess Android envelope.

## Method D — low-dimensional/factorized bridge

Deferred. A factorized bridge could reduce parameters, but adding architectural complexity before establishing the dense linear baseline would confound interface, optimization, and representation findings. The dense bridge is already only 221,760 parameters and bridge-only training measured below the RSS ceiling.

## Resource boundary

Bridge-only optimization was measured separately from MiniLM feature extraction and decoder inference. The bridge-only process peaked at 38,260,736 bytes. MiniLM extraction peaked at 892,514,304 bytes in its own process, and decoder inference peaked below 256 MiB in its own process. Therefore the bridge is feasible as an isolated stage, but whole-pipeline compliance is not established.
