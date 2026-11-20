# LITE-007G Evaluation Controls

Controls are classified as implemented, evaluated, or pending. No control is silently substituted for another.

| ID | Control | Implemented | Actually evaluated in LITE-007F/G | Notes |
|---|---|---:|---:|---|
| A | Lexical retrieval | pending | no | Requires explicit token/index contract and equivalent candidate budget |
| B | Direct cosine retrieval | yes | yes | LITE-007F direct query/evidence cosine; not trained |
| C | Hybrid lexical + semantic | pending | no | Must combine independently defined lexical and cosine scores |
| D | Untrained neural memory | yes | yes | Paired validation baseline |
| E | Scoring-head-only neural memory | yes | yes | LITE-007E checkpoint |
| F | Memory-projection-only neural memory | yes | yes | LITE-007F checkpoint |
| G | Joint score head + memory projections | yes | yes | Parameter combination evaluated; no LITE-007G updates |
| H | Trained direct query/evidence scorer without slots | pending | no | Must be a separately trained scorer; cosine is not a substitute |
| I | Query-independent memory slots | pending | no | Requires explicit ablation with query-conditioning removed |
| J | Query-conditioned associative memory | yes | yes | Existing architecture; projection-only training evaluated |

## LITE-007F audit

The current corpus contains at most five eligible records per scenario. Consequently Recall@5 is exactly 1.0 for all reported associative controls and cannot distinguish ranking quality. The eight-update episodes also touch only a tiny deterministic prefix and are exposure probes, not a meaningful curriculum.

The direct cosine control substantially outperformed the neural controls because the fixture's query embeddings already encode direct semantic correspondence, while the untrained associative path compresses evidence through random projections, attention, residual refinement, and a max-slot score head. This is a hypothesis about the fixture and architecture bottleneck, not proof that the architecture cannot learn.

The previous report did not include abstention accuracy, unsupported retrieval, temporal correctness, or policy-violation denominators. These metrics are required in the next evaluation implementation. Answerable ranking examples must be separated from insufficient-evidence examples.
