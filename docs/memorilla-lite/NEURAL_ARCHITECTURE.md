# LITE-006 Neural Architecture

Status: bounded architecture implementation and mathematical validation complete. No optimizer update, epoch, checkpoint selection, or full neural training was performed in LITE-006. LITE-007 separately wires the previously unused top-level memory projections into masked memory-summary slot initialization without changing the 429,312-parameter target.

## Architectural target

The module accepts one frozen 384-D query embedding, an eligible collection/stream of 384-D memory embeddings, stable evidence identities supplied outside the module, and a validity mask. It emits exactly 8 query-conditioned slots of 384 dimensions regardless of collection size.

Initial configuration:

```text
input embeddings:       384
latent dimension:       128
slots:                  8
attention heads:        4
cross-attention blocks: 1
feed-forward dimension: 256
output dimension:       384
score projection:       64
parameters:             429,312
raw Float32 weights:    1,717,248 bytes
initialization seed:    20261120
```

The parameter target is below 500,000. Parameters are random/untrained and must not be interpreted as cognitive capability.

## Forward path

```text
frozen query [384]
  -> query projection [128]
  -> learned slots [8,128] + broadcast query conditioning
  -> pre-normalized 4-head cross-attention over eligible memory embeddings
  -> residual update
  -> pre-normalized GELU(tanh approximation) FFN 128 -> 256 -> 128
  -> residual update
  -> output projection [128 -> 384]
  -> query-conditioned slots [8,384]
```

Masks are applied before softmax. Empty or fully masked collections skip cross-attention and return finite query-conditioned slots after the bounded refinement path. Slot count never depends on record count.

## Streaming attention

The NumPy reference implements collection-wide two-pass online softmax:

1. Pass one computes per-slot/head logit maxima over all valid chunks.
2. Pass two accumulates exponentiated weighted values and denominators.
3. The result is normalized globally, not independently per chunk.

Full versus streamed agreement was below `6.5e-10` maximum absolute error at 100, 1,000, 5,000, and hand-constructed masked collections. No mandatory top-k prefilter is used.

## Evidence policy boundary

The neural module does not authorize evidence. Collection, access-scope, model, preprocessing, and historical cutoff restrictions are applied before neural scoring by `policy_rank`. Only eligible evidence IDs enter ranking. Every selected ID is passed through the existing authoritative `EvidenceVerifier` and source-hash resolver. Unverified candidates are omitted or produce abstention.

Attention weights are intermediate numerical values only. They are not explanations, factual proof, or verified evidence.

## Evidence scoring head

The compact NumPy scoring interface:

- Projects slots and evidence embeddings into a compact score space.
- L2-normalizes both representations.
- Uses maximum slot-to-evidence similarity as the evidence score.
- Applies validity/policy masks before ranking.
- Sorts deterministically by descending score and evidence ID.
- Supports multiple positives by returning independently ranked evidence IDs.
- Returns explicit empty results for empty collections.

This is a Mode B evidence-selection interface, not a language model or text-generation bridge. Diversity is prepared as a separate regularization term rather than assumed from attention weights.

## Native scoring-head preparation path

The first verified native learning path keeps the encoder, memory slots, query-conditioning projections, attention, feed-forward layers, and output projections frozen. Only `score_slot_w`, `score_slot_b`, `score_mem_w`, and `score_mem_b` are updated by plain Float32 SGD. When eager corpus assembly is too slow for Android, `lite.training.prepared_dataset` creates a deterministic train-only indexed artifact from the existing verified embeddings. The artifact is content-addressed to the source dataset, embedding archive, manifest, and encoder contract; it contains no holdout data and is loaded one example at a time. This preparation layer is an execution optimization, not a change to evidence eligibility or training semantics.

## Training objective preparation

The primary proposed objective is multi-positive pairwise logistic ranking: each authoritative positive evidence ID is compared against valid negatives. Optional terms are hard-negative weighting, multi-evidence coverage, slot diversity regularization, and evidence-selection abstention. Labels must resolve to authoritative evidence IDs and may not originate from future, unauthorized, or sealed holdout records.

## Upstream relationship

The upstream implementation inspected in `memorilla/memory.py` is Apache-licensed and directly establishes these principles:

- learned memory-query/slot initialization;
- query-conditioned slot processing;
- multi-head cross-attention from fixed slots to document embeddings;
- residual and normalization paths;
- feed-forward refinement;
- fixed output token count independent of document count;
- optional collection-level key/value caching.

LITE-006 is not an exact reproduction. Newport River adaptations are:

- 8 slots instead of upstream default 16;
- 128 latent width instead of operating at the full embedding width;
- 4 heads and one block for Android resource bounds;
- 384-D output to remain compatible with the existing frozen encoder;
- NumPy full/streamed reference independent of PyTorch runtime;
- explicit authoritative evidence-policy gate outside the model;
- no decoder, vLLM, language bridge, retrieval initialization, or training loop;
- safe non-executable weight serialization with checksums and encoder compatibility.

Upstream source and Apache licensing/attribution remain preserved. The new `lite/neural` implementation is a Newport River mobile adaptation and does not claim Stanford model equivalence.
