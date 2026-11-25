# LITE-008R teacher-student protocol

Status: BLOCKED by failed gradient qualification. No teacher embeddings were generated and no student optimizer updates were attempted.

If authorized after qualification, use only training-partition facts, separately count teacher generation and student updates, verify teacher preference with frozen-decoder likelihoods, and never use development oracle embeddings. Embedding MSE alone will not be treated as semantic learning.
