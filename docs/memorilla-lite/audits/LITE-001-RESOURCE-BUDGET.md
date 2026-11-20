# LITE-001 Resource Budget

## Measured normal-operation baseline

Three low-load snapshots were collected 30 seconds apart. No stress test or inference ran during collection.

- Kernel-visible physical RAM: 5,528,084 KiB (~5.27 GiB).
- Available RAM: 1,370,360 -> 1,339,296 -> 1,339,468 KiB (~1.31 -> 1.28 -> 1.28 GiB).
- Free RAM: 254,992 -> 134,304 -> 119,380 KiB.
- Swap configured: 8,388,604 KiB (~8 GiB).
- Swap free: 5,496,912 -> 5,536,960 -> 5,540,288 KiB (~5.24 -> 5.28 -> 5.28 GiB).
- Storage available: 6.9G on the measured Termux filesystem view, 94% used.
- CPU load: unavailable because `/proc/loadavg` is permission-denied.
- Thermal readings: unavailable; no readable thermal zones.
- Battery telemetry: unavailable; `termux-battery-status` is not installed/accessible.

## Provisional LITE-002 budget

These are conservative engineering limits derived from measured availability, not measured model performance:

| Workload | Initial budget | Rationale/status |
|---|---:|---|
| Encoder process RSS | <=256 MiB target | provisional; leaves room from ~1.28 GiB available |
| Attention process RSS | <=256 MiB target | provisional; must be measured with fixture workload |
| Combined encoder + attention | <=512 MiB target | provisional ceiling for first prototype |
| Derived vector storage | <=512 MiB initial | storage headroom is only 6.9G at 94% used |
| Optional decoder | not budget-approved | no local model/runtime and limited available RAM |
| Concurrent inference processes | 1 | fail-safe until thermal/load telemetry exists |
| Model/data download | 0 for LITE-001 | prohibited in this milestone |

The current available RAM is not a guarantee that a 512 MiB process will be safe; Android reclaim and background restrictions can intervene. These budgets must be revised only after LITE-002 measured smoke tests.

## Acceptance implications

The first runtime path should be CPU-only, single-process, short-lived or owner-launched, with deterministic fallback. No continuous worker pool, sustained inference loop, or local decoder should be enabled from this audit. Thermal and battery telemetry are missing and must remain explicit blockers for endurance acceptance.
