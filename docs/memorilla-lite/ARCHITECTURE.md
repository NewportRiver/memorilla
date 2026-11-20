# Memorilla-Lite Android Architecture

## Authority boundary

APEX authoritative SQLite records and content-addressed artifacts remain the source of truth. Memorilla-Lite consumes sanitized snapshots and emits hypotheses/candidates that must resolve back to exact evidence. Derived vectors, caches, clusters, and learned weights are rebuildable and never authorize tools, policy, brokerage, or production changes.

## Components

1. Snapshot exporter: deterministic, sanitized, hash-verified records.
2. Frozen encoder: initially 384-D ONNX Runtime ARM64 candidate.
3. Derived store: artifact ID, source hash, encoder revision, chunk revision, quantized vector, scale, cluster metadata.
4. Memory module: query-conditioned slots, initially 8 slots and one compact refinement block.
5. Evidence resolver: exact IDs, artifact hashes, provenance and integrity failures.
6. Optional bridge: bounded textual output first; raw embedding prefix only as an experimental matched-decoder path.

## Modes

- Mode A: cosine/top-k baseline; control only.
- Mode B: learned slots returning evidence IDs/text; first scientific target.
- Mode C: learned slots plus bounded local text bridge.
- Mode D: 1,024-D soft-token prefix into matched Qwen3-0.6B; research-only until proven.

Mode B is the first target. Mode D is not equivalent to Mode A or conventional RAG and must be separately measured.

## Data and trust flow

```text
Authoritative APEX records -> sanitized immutable snapshot -> encoder -> derived vectors
query -> query encoder -> learned slots -> evidence candidates -> hash-verified resolver
                                                    \-> optional decoder bridge
```

Source-hash mismatch invalidates derived data. Missing/corrupt evidence is a hard failure with deterministic fallback. Credentials and private mission histories are excluded before encoding.

## Resource strategy

Keep full-store attention for small corpora and measure the crossover. Add hierarchical routing only after comparing missed-evidence rate against full attention. Keep the decoder optional and isolated from the evidence store. Android thermal and process constraints are first-class acceptance signals.

## Upstream preservation

Do not rewrite Stanford’s upstream algorithm in place. Android-specific adapters and experiments belong in a clearly separated mobile profile. Preserve upstream attribution, license files, model provenance, and revision hashes.
