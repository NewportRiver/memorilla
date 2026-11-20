# LITE-001 Runtime Audit

## Existing Termux inventory

| Capability | Result | Compatibility status |
|---|---|---|
| Python | 3.13.13 at `$PREFIX/bin/python` | present |
| pip | 26.1.2 | present |
| C compiler | Clang 21.1.8; gcc/g++ wrappers present | present |
| CMake | command absent | blocker for source builds |
| Ninja | command absent | optional build blocker |
| Git | 2.54.0 | present |
| SQLite CLI | command absent | Python SQLite available |
| SQLite library | 3.53.1 | present |
| SQLite FTS5 | `sqlite_compileoption_used('ENABLE_FTS5')=1` | verified through Python |
| NumPy | 2.4.4 ARM64 package | present |
| PyTorch | 2.11.0 ARM64 package | present; not selected as mobile runtime yet |
| ONNX Runtime | absent | not available without installation |
| llama.cpp Python/CLI/server | absent | not available |
| ONNX Python package | absent | not available |
| Transformers | absent | not available |
| sentence-transformers | absent | not available |
| Existing `.onnx` models | none found under home | no local model |
| Existing `.gguf` models | none found under home | no local model |

No package was installed or upgraded.

## Runtime recommendation

LITE-002 should not begin with a model download. After owner authorization, the preferred path is a bounded native Termux feasibility check for an ARM64-compatible ONNX Runtime route, but the current environment lacks ONNX Runtime, CMake, and Ninja. The first implementation decision should therefore be made after checking whether a compatible prebuilt Termux package is available and measuring its footprint.

If native Termux ONNX Runtime is unavailable or incompatible, use a small isolated C/C++ or Android-native proof rather than changing APEX or installing a large Python stack. `llama.cpp` remains a later decoder experiment and is not present today.

Python SQLite with FTS5 is usable now for metadata/search fixtures. NumPy is usable now for bounded numerical fixtures. PyTorch is present but its presence alone is not evidence that the proposed training or mobile inference workload fits the device.

## Limitations

The audit did not install, upgrade, compile, download, or execute any inference runtime. Compatibility with ONNX Runtime Android, ExecuTorch, ncnn, or `llama.cpp` on this exact handset remains unverified.
