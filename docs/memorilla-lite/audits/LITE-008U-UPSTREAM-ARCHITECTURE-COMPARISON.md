# LITE-008U upstream architecture comparison

## Verified upstream details

The locally retained upstream `memorilla/memory.py` and README establish a frozen decoder plus frozen document/question encoder endpoints, learned query-conditioned memory slots, multi-head cross-attention over the full document collection, residual and LayerNorm paths, FFN refinement, fixed output token count, and optional collection-level key/value caching. The README documents 16 memory tokens, Qwen3-8B decoder, Qwen3-Embedding-4B encoder, staged supervised answer reconstruction, and million-example training recipes.

The upstream recipe documents Stage 1 reconstruction of approximately two million Wikipedia passages, Stage 2 continuation on approximately 1.34 million QA/summarization examples, Stage 3 multitask training, AdamW, warmup/cosine schedule, gradient clipping and bfloat16 mixed precision. These are upstream documentation facts, not Lite measurements.

## Lite differences

Lite uses a frozen 384-D MiniLM encoder, one 576-D latent slot, a residual linear projection, no query feature in the bridge, no learned attention, no collection-wide memory module, no document self-attention, no FFN refinement, no supervised pretraining curriculum, and a 135M SmolLM2 frozen decoder. Lite’s direct bridge experiments use tiny synthetic counterfactual datasets and bounded CPU updates.

## Proposed adaptations

Teacher-guided alignment from individually optimized P embeddings is a Newport River experimental adaptation. It is not an upstream implementation detail and does not replace upstream query-conditioned attention, multi-slot memory transformation or answer-loss supervision.
