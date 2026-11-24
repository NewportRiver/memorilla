# LITE-008D-R3 GO / NO-GO Decision

## Decision: CONDITIONAL GO

The Android decoder qualification remains plausible, but it cannot proceed because the downloaded official model artifact does not match the supplied expected SHA-256. The model size and pinned revision match; content identity is unresolved.

## Evidence

- Baseline `6e9e1df` and branch verified.
- CMake `4.3.4`, clang `21.1.8`, and Make available.
- Pinned llama.cpp checkout verified at `969d7333172c522d6a60fffb8dd35288ce8df236`.
- Official model revision verified: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Model size matched: `269,060,552` bytes.
- Supplied expected SHA-256 did not match actual local content SHA-256.
- No conversion dependency was installed.
- No build, quantization, or decoder execution occurred.
- MiniLM and representative neural checkpoints remained intact.
- Nine pending X11 packages remained untouched.
- No SQLite or holdout access occurred.

## Why not GO

GO requires verified model identity and measured deterministic generation within the approved resource envelope. Neither model identity nor any decoder execution result is currently sufficient for that claim.

## Why not NO-GO

The mismatch may be a provenance-labeling error: the supplied value is consistent with previously observed remote ETag metadata, while the local content hash is different. The pinned revision and expected byte size match. This is an unresolved verification blocker, not a demonstrated model/runtime failure.

## Required resolution

Provide or authorize one of:

1. an authoritative content SHA-256 for the exact pinned `model.safetensors`; or
2. explicit owner acceptance that the supplied value is an ETag/non-content identifier, together with an independently verified artifact provenance rule.

After resolution, re-hash the existing file or reacquire it, then resume with conversion dependency inspection. Do not use the current mismatch as a basis for conversion.

## Scientific boundaries

- Gate D: not passed.
- Gate E: not passed.
- Gate F: not passed.
- No latent-memory projection or injection.
- No training or fine-tuning.
- No retrieval or encoder changes.
- No decoder qualification claim.

Stop at owner review.
