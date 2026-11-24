# LITE-008D-R3 Dependency Manifest

## Status: no new conversion dependencies installed

The process stopped at official model checksum verification before inspecting or installing Python conversion dependencies. Existing packages were preserved.

### Existing native tools

- CMake: `4.3.4`, ARM64 Android executable
- clang: `21.1.8`
- make: present
- llama.cpp source: pinned detached commit `969d7333172c522d6a60fffb8dd35288ce8df236`

### Existing Python modules before R3

- `torch`: present
- `onnxruntime`: present
- `transformers`: unavailable
- `safetensors`: unavailable
- `huggingface_hub`: unavailable
- `sentencepiece`: unavailable
- `protobuf`: unavailable

No Python virtual environment was created and no package was installed. Because the pinned model was not accepted for conversion, the conversion script's actual import requirements were not resolved speculatively.

### Native package state

CMake recovery packages remain installed from R2:

- `cmake 4.3.4`
- `libarchive 3.8.8`
- `jsoncpp 1.9.8`
- `libuv 1.52.1`
- `rhash 1.4.6-1`

The nine pending X11 packages remain untouched and unconfigured. No upgrade, removal, forced operation, repository change, or Python installation occurred during R3.

## Dependency decision

No additional dependency transaction is justified until the model checksum conflict is resolved. This avoids installing conversion packages for an artifact that is not yet accepted for use.
