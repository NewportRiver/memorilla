# LITE-008 — Latent token injection

## Objective
Test raw embedding prefix with a pinned Qwen3-0.6B/llama.cpp path.

## Scientific rationale
Raw embedding API mechanism is reported plausible, but Qwen3 Android integration is unverified.

## Dependencies
LITE-006; LITE-007; pinned decoder.

## Implementation steps
Desktop one/eight-vector proof; then Android deterministic proof; compare text bridge.

## Tests
Protocol, shape, deterministic output, fallback, OOM and unsupported-path tests.

## Acceptance criteria
Explicitly classify supported/unverified; no claim before both environments pass.

## Resource limits
No large download without owner approval; decoder optional.

## Rollback
Remove binary/model artifacts; disable experimental route.

## Completion reporting
Report exact revision, model hash, results and failures.

## Status
Not started. Owner authorization required before execution.
