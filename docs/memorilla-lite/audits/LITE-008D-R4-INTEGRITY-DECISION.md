# LITE-008D-R4 Integrity Decision

## Decision: PASSED

The official SmolLM2-135M model integrity gate passed after resolving the identifier-classification error from R3.

The previously recorded `8595fbb3...` value is the Xet hash and HTTP ETag, not the payload SHA-256. The authoritative Git LFS content SHA-256 is `80521b4028...`, which matches the locally calculated hash exactly.

### Acceptance checks

- Official repository identity: passed
- Exact pinned revision: passed
- Exact file path: passed
- Expected size `269,060,552`: passed
- Authoritative content SHA-256: passed
- Configuration and tokenizer revision consistency: passed
- Existing local artifact retained: passed

The historical R3 manifest remains unchanged. R4 records the corrected provenance without silently rewriting prior evidence.

## Qualification result

The corrected integrity gate permitted the previously approved ordinary decoder protocol to resume.

- Pinned llama.cpp commit: `969d7333172c522d6a60fffb8dd35288ce8df236`
- ARM64 CPU-only build: passed
- Q4_K_M conversion: passed
- Three deterministic repeats per tested context: passed
- Effective 256-token context: passed
- Effective 512-token context: passed
- Requested 128-token context: accepted by runtime as effective 256-token context
- Maximum observed decoder RSS: `198,004,736` bytes
- RSS ceiling: `268,435,456` bytes

## Final R4 status

GO for ordinary decoder generation and repeatability within the approved resource limits.

This is an infrastructure qualification result only. Gate D, Gate E, and Gate F remain unpassed. Latent-memory integration requires separate authorization.
