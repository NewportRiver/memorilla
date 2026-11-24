# LITE-008N Regression Recovery Proposal

No dependency installation is authorized or performed.

Current verified scope:

- Targeted LITE suite: run sequentially in the available environment.
- Python compilation, JSON parsing and diff integrity: independent checks.
- Complete repository regression: currently blocked at collection because system Python lacks `transformers`; the qualified model environment lacks `datasets`.

Minimal recovery proposal:

1. Use an owner-provided offline environment or existing package cache containing the repository-declared test dependencies.
2. Do not alter the Android system package state or install into the active runtime as part of this milestone.
3. Run `python -m pytest -q` from that isolated environment after dependency provenance is recorded.
4. Report collection failures separately from test failures and retain the current targeted result as a distinct scope.
5. Verify official artifact hashes and no persistent experimental process after the recovery run.
