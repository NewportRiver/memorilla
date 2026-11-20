# LITE-001 Android Hardware Audit

## Scope
Read-only audit on the existing Android device. No root access, package changes, model downloads, stress testing, or service interruption occurred. Three snapshots were collected at 21:20:00, 21:20:30, and 21:21:00 local time.

## Measured identity

| Field | Observation | Status |
|---|---|---|
| Model | `SM-A376U` | measured |
| Android | `17` | measured |
| SDK | `37` | measured |
| ABI | `arm64-v8a` | measured |
| Kernel | `6.1.162-android14-11` | measured |
| SoC manufacturer | Samsung | measured |
| SoC model property | `s5e8845` | measured |
| Logical CPU count | 8 | measured |
| CPU features | ARM crypto/NEON-related features including `asimd`, `aes`, `sha1`, `sha2`, `crc32`, `atomics` | measured |

The device reports eight logical CPUs. The exact physical core topology was not inferred from the model property.

## Memory and storage

`/proc/meminfo` reported `MemTotal=5,528,084 KiB`, approximately 5.27 GiB kernel-visible physical memory. This is not a claim about the handset’s retail RAM SKU.

| Snapshot | MemAvailable | MemFree | SwapTotal | SwapFree |
|---:|---:|---:|---:|---:|
| 1 | 1,370,360 KiB | 254,992 KiB | 8,388,604 KiB | 5,496,912 KiB |
| 2 | 1,339,296 KiB | 134,304 KiB | 8,388,604 KiB | 5,536,960 KiB |
| 3 | 1,339,468 KiB | 119,380 KiB | 8,388,604 KiB | 5,540,288 KiB |

The measured filesystem view reported 105G total, 98G used, and 6.9G available at 94% utilization. This is the available filesystem visible from the measured Termux path, not a claim about total physical storage SKU.

Physical RAM and swap are distinct: approximately 5.27 GiB kernel-visible RAM versus 8 GiB configured swap. Swap availability must not be counted as equivalent model RAM.

## CPU, thermal, battery, and process restrictions

- `/proc/loadavg`: permission denied; idle load was not fabricated.
- Readable thermal sysfs zones: 0.
- `termux-battery-status`: unavailable.
- Android property `persist.sys.fflag.override.settings_enable_monitor_phantom_procs=false` was observed, but direct background/phantom-process policy verification was unavailable.
- No stress or sustained inference was performed.

## Feasibility interpretation

The device is ARM64-capable and has available native CPU/NEON/crypto support, but the measured available memory was approximately 1.34 GiB and storage headroom was only 6.9G. A compact CPU-first encoder and learned attention prototype may be feasible within a strictly bounded envelope. A large local decoder is not accepted by this audit and requires a later measured, authorized experiment.
