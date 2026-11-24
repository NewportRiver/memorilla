# LITE-008F Architecture Decision

## Selected method

Proceed with Method B only: a standalone dense linear `384 → 576` projection with bias, trained against genuine Q4 decoder input-embedding targets.

## Frozen state

- MiniLM encoder weights and tokenizer
- SmolLM2-135M Q4_K_M decoder weights
- llama.cpp runtime
- Existing Memorilla neural-memory module and checkpoints
- SQLite evidence and hybrid retrieval architecture

## Trainable state

Only the bridge `W[384,576]` and `b[576]`, totaling 221,760 Float32 parameters. No `MemoryModule` tensor receives gradients. No decoder parameter receives gradients.

## Objective

For normalized MiniLM-derived representation `x_i` and target `t_i` equal to the mean genuine Q4 decoder input embedding for a synthetic answer string:

`L = mean_i,j ((x_i W_j + b_j) - t_ij)^2`.

This is explicitly an embedding-reconstruction objective, not answer likelihood and not proof of semantic conditioning.

## Data and acceptance

Use 24 synthetic entity/value examples split by entity: 16 train and 8 development. Use fixed seed `20261124`, 300 SGD steps, learning rate `0.08`. Accept bridge feasibility only if loss decreases on the disjoint development entities, finite-difference checks pass on selected parameters, serialization round-trips, and bridge-only RSS remains below 268,435,456 bytes. Semantic-conditioning success is not accepted from MSE alone.

## Decision

This is the smallest scientifically defensible bridge experiment. It avoids unverified decoder backpropagation, high-dimensional black-box search, production architecture changes, and complete associative-memory training.
