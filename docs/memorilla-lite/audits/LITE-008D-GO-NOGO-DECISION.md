# LITE-008D GO / NO-GO Decision

## Decision: CONDITIONAL GO

The official SmolLM2-135M decoder was not shown to execute on Android because the qualification stopped at a missing, unauthorized build dependency. This is not a decoder failure and not a successful qualification.

## Evidence

- Baseline `4ae5245` and branch were verified.
- Working tree was clean before report generation.
- MiniLM checksum matched the pinned manifest:
  `4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474`.
- Representative LITE-007L checkpoints were hashed and unchanged:
  - associative: `fb9a273fd96bfe9d6cf35da2145e45492b414ad6a02cd35acfba999886587a7a`
  - direct: `a9e190cd7a0c81dc4ac070b9299819fced77d3b38432c9534577b8b19691d2c8`
- No SQLite artifact was opened or modified.
- No holdout data was accessed.
- clang `21.1.8` and Python `3.13.13` are present.
- CMake is absent.
- No model, tokenizer, llama.cpp checkout, dependency, build, conversion, or execution occurred.

## Why not GO

GO requires measured ordinary generation within the 256 MiB RSS limit, deterministic repeated output, and resource telemetry. All decoder measurements are unavailable because the build cannot begin.

## Why not NO-GO

The selected decoder has not failed an Android execution test. The blocker is a resolvable authorization/dependency gap: CMake is missing and installation was not authorized. No unapproved package, runtime, model, quantization, or source commit was substituted.

## Required next authorization

Authorize a separately bounded dependency step specifying a pinned Termux CMake package/version, storage cost, and rollback method. After that approval, repeat preflight before any acquisition. If the dependency step passes, acquire only the pinned official model and llama.cpp commit, then execute the ordinary-generation protocol.

## Gates and boundaries

- Gate D: not passed.
- Gate E: not passed.
- Gate F: not passed.
- No latent-token projection or injection occurred.
- No neural-memory architecture or retrieval baseline changed.
- No training or holdout access occurred.

This milestone remains stopped at owner review. No LITE-008E work is authorized by this report.
