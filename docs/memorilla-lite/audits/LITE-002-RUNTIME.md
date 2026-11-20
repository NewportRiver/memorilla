# LITE-002 — Minimal Runtime and Computational Feasibility Audit

Status: completed for the bounded NumPy/Python/SQLite reference path; ONNX installation and model work remain deferred.

## Scope and controls

- Repository baseline: `76e5348a69875c5bcc714d842e748ecc3586bdf4`.
- Synthetic data only.
- No packages installed or upgraded.
- No models downloaded.
- No training, production APEX modification, brokerage activity, root access, cloud expense, or endurance testing.
- One-process benchmark execution with maximum two numerical threads requested.
- Per-process timeout target: 30 seconds.
- Temporary databases and benchmark fixtures were created outside Git and removed by temporary-directory cleanup.

## Revised runtime strategy

The LITE-001 audit observed approximately 1.28 GiB available RAM and 6.9G available storage on the measured Termux filesystem. The initial reference runtime therefore uses existing NumPy, Python, and SQLite rather than making ONNX Runtime an immediate prerequisite.

ONNX Runtime was investigated through the existing Termux package index without installation. Native ARM64 packages are listed:

```text
onnxruntime          1.27.0
aarch64
python-onnxruntime   1.27.0
aarch64
llama-cpp            0.0.0-b9590-0
aarch64
cmake                4.3.4
aarch64
```

The installed Python is Android/Termux-linked through `/system/bin/linker64`, not a generic glibc wheel environment. A future ONNX test should therefore prefer the native Termux package path or an Android-native/C++ build, subject to owner authorization and storage/RAM preflight. No compatibility claim is made until installed and executed.

PyTorch was not imported until after a successful resource preflight. It was then tested only in a subprocess with two threads and a 30-second timeout.

## Resource protection

Configured provisional limits:

| Limit | Value | Enforcement status |
|---|---:|---|
| Minimum available RAM | 1 GiB | preflight check |
| Abort available RAM | 768 MiB | preflight policy; no test reached it |
| Target incremental RSS | 128 MiB | observation target only |
| Maximum RSS | 256 MiB | observed/recorded; not kernel-enforced |
| Maximum new storage | 50 MiB | preflight policy |
| Numerical threads | 2 | environment and PyTorch setting |
| Concurrent processes | 1 | single benchmark process |
| Timeout | 30 seconds | subprocess test enforcement |
| Sustained swap growth | prohibited | no endurance test performed |

The platform does not provide reliable user-space hard RSS enforcement in this harness. RSS values are observations, not guarantees. Swap is recorded separately and is not counted as physical RAM. Thermal monitoring remains unavailable, so no endurance test was performed.

## Smoke-test results

The bounded smoke suite validated:

- Python execution.
- NumPy float32 matrix multiplication.
- NumPy float16 matrix multiplication.
- INT8 storage and explicit dequantization.
- SQLite transactions.
- SQLite BLOB persistence.
- SQLite FTS5 retrieval.
- SHA-256 evidence identity.
- Subprocess timeout and cleanup.
- Runtime telemetry collection.

Targeted tests:

```text
9 passed in 0.75s
```

Observed smoke process telemetry:

```text
Preflight available RAM:  1,373,040,640 bytes
Peak RSS:                 34,664,448 bytes
RSS at snapshot:          34,664,448 bytes
Threads:                  1
Swap total:               8,589,930,496 bytes
Swap free after smoke:    5,436,137,472 bytes
CPU load:                 unavailable; /proc/loadavg permission denied
```

Numerical smoke errors:

```text
float16 matrix max absolute error: 0.010982513427734375
INT8 dequantization max error:     0.01558944582939148
```

## Bounded PyTorch probe

After preflight reported available RAM above the 1 GiB minimum, a child process executed:

```text
import torch
threads=2
x=torch.ones((256,256),dtype=torch.float32)
y=x@x
```

Result:

```text
Return code: 0
PyTorch:     2.11.0
Output:      (256, 256), value 256.0
Peak RSS:    221,339,648 bytes
Timeout:     no
```

This is below the 256 MiB observation ceiling but above the 128 MiB target. It is a feasibility probe only, not an approval for training or large-model inference. PyTorch remains unsuitable as the first mobile runtime until workload-specific measurements exist.

## Limitations

- `/proc/loadavg` is inaccessible.
- Thermal and battery telemetry remain unavailable.
- RSS is observed, not hard-enforced.
- No ONNX Runtime binary was installed or executed.
- No local model exists.
- No sustained or endurance workload was run.
- No result demonstrates learned associative memory or Stanford benchmark performance.

The NumPy reference is mathematical infrastructure, not a trained cognitive system.
