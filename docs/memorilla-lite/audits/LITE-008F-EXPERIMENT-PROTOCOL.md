# LITE-008F Experiment Protocol

## Synthetic data

Twenty-four randomized entity/value templates were generated as `entity-00` through `entity-23`, with values `value-00` through `value-97` selected by a deterministic mapping. Entities 00–15 are training identities; entities 16–23 are development identities. No protected holdout was accessed.

MiniLM input text was: `Question: What is the value for {entity}? Memory: {entity} is associated with {value}.` Target text was the corresponding value string. MiniLM outputs were attention-mask mean pooled and L2 normalized by the existing frozen encoder contract. Target vectors were means of genuine Q4 `token_embd.weight` rows for the target tokenization.

## Primary run

- Bridge: dense Float32 linear layer with bias
- Seed: `20261124`
- Steps: 300
- Optimizer: full-batch SGD
- Learning rate: `0.08`
- Objective: target-embedding MSE
- Train/dev split: 16/8 entities
- Decoder: frozen Q4_K_M SmolLM2-135M
- Runtime: pinned llama.cpp `969d7333172c522d6a60fffb8dd35288ce8df236`
- Decoder contexts: 256 for the bounded downstream probe
- CPU-only, one thread, one subprocess per decoder run
- RSS ceiling: 268,435,456 bytes

## Verification

A finite-difference check used eight selected bridge weights with `h=1e-3`. The bridge was serialized and loaded back with checksum verification. The projected output was passed to the LITE-008E native harness and compared with a zero-embedding control. This downstream probe is directional interface evidence only; it is not an answer-accuracy experiment.
