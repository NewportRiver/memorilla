# LITE-003 — Lightweight semantic encoder

## Objective
Benchmark 384-D encoder candidates on sanitized synthetic fixtures.

## Scientific rationale
MiniLM/BGE candidates preserve a small embedding dimension but require new memory training.

## Dependencies
LITE-001; LITE-002.

## Implementation steps
Pin model revisions/licenses; encode fixtures; compare latency, RSS, dimension, determinism.

## Tests
Hash inputs/outputs; repeatability and resource tests.

## Acceptance criteria
Candidate chosen by measured accuracy/resource report.

## Resource limits
No private APEX history; bounded fixture size.

## Rollback
Remove derived model/cache; retain manifest.

## Completion reporting
Report candidate decision and rejected alternatives.

## Status
Completed with the native Termux ONNX Runtime 1.27.0 path and one verified `all-MiniLM-L6-v2` ARM64 quantized artifact. Actual text produced finite normalized 384-D embeddings; 13 targeted tests passed; bounded benchmarks passed at 16/32/64/128-word inputs; provisional semantic top-2 criteria passed 3/3 queries; encoder-to-attention compatibility passed. ONNX upstream Android support warning, unavailable thermal/battery telemetry, and the pre-existing dpkg configuration blocker remain documented. LITE-004 requires separate owner authorization.
