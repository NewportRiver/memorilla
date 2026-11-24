# LITE-008D-R4 Determinism Results

## Result: passed

The fixed ordinary-generation experiment was run three times for each requested context setting.

Command contract:

- Q4_K_M GGUF
- CPU-only, `-ngl 0`
- One generation thread and one batch thread
- Prompt: `The capital of France is`
- 32 requested output tokens; 31 generated runs observed
- Temperature `0`, top-k `0`, top-p `1`, min-p `0`
- Greedy sampler chain
- Seed argument `42`
- 120-second timeout
- Periodic RSS monitoring with process termination above `268,435,456` bytes

All nine runs exited with status `0`, stayed below the RSS ceiling, and produced the identical continuation:

`the capital of the country. It is the largest city in France and the second largest in Europe. It is located in the south of France, in the department`

### Context sweep

| Requested | Effective | Repeats | Max RSS | Generation rate |
|---:|---:|---:|---:|---:|
| 128 | 256 | 3/3 | 191,946,752 bytes | 78.58–79.12 tok/s |
| 256 | 256 | 3/3 | 191,942,656 bytes | 79.40–80.03 tok/s |
| 512 | 512 | 3/3 | 198,004,736 bytes | 79.35–79.95 tok/s |

The runtime normalized the requested 128-token context to an effective 256-token context. This is recorded as an execution limitation; it is not represented as a successful distinct 128-token qualification.

### Additional measurements

- Maximum observed decoder RSS: `198,004,736` bytes
- RSS ceiling: `268,435,456` bytes
- Sampled CPU utilization: `98.0%` of one CPU
- Temperature: unavailable because no readable thermal sensor was present
- KV-cache allocation: unavailable as a separate runtime metric
- OOM or forced termination: none

The runtime log reported seed `4294967295` despite the seed argument. Because temperature-zero greedy output was identical across all repeats, repeatability passed; the seed-reporting discrepancy is preserved rather than ignored.

## Boundary

This is ordinary decoder infrastructure qualification. It does not test latent-memory tokens, the 384-to-576 projection, learned memory, retrieval, or any scientific effectiveness gate.
