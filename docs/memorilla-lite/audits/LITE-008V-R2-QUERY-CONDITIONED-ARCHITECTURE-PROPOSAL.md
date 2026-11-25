# Query-Conditioned Architecture Proposal

R2 did not implement or train a new architecture. The result removes one narrow objection: the current linear map can interpolate the eight corrected P calibration targets because X has rank 8 and the target system is consistent. It says nothing about reusable capacity.

For a future Android-compatible comparison, review LITE-006/LITE-007 components and upstream Memorilla's query-conditioned memory path. The smallest candidate should retain one latent slot and frozen evidence/query embeddings, then add only a compact query-conditioned residual transform. A bounded design would use a low-rank query gate or a single-head compact attention block followed by a 384-to-576 projection, with residual connection and decoder-scale initialization. Parameter count, RSS and sequence-cost estimates must be measured in a separate authorized milestone; no estimates are promoted here as measured facts.

The proposal must remain distinct from full associative-memory training and must pass exact objective/decoder numerical qualification before any optimizer update.
