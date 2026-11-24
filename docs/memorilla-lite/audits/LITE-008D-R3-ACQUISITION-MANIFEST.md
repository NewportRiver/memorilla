# LITE-008D-R3 Acquisition Manifest

## Status: stopped at model integrity gate

The pinned llama.cpp source and official model artifacts were acquired into isolated cache paths. Qualification stopped before build, conversion, or execution because the supplied expected SHA-256 for `model.safetensors` did not match its actual content hash.

### Repository and runtime

- Project branch: `feature/memorilla-lite-android`
- Required baseline: `6e9e1df`
- llama.cpp source: `$HOME/src/lite008d-llama.cpp`
- Source commit: `969d7333172c522d6a60fffb8dd35288ce8df236`
- Detached checkout verified with `git rev-parse HEAD`

### Official model

- Repository: `HuggingFaceTB/SmolLM2-135M`
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- License: Apache-2.0
- Model file: `$HOME/.cache/memorilla-lite/LITE-008D-R3/model/model.safetensors`
- Actual size: `269,060,552` bytes — matches the required size
- Supplied expected SHA-256: `8595fbb3d6026b3818e1a9d538e145d63abbdb21e948de9b4968b5a16e96a579`
- Actual content SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- Result: checksum mismatch; artifact not accepted for conversion

The supplied expected value matches the previously observed Hub ETag metadata, but it was not treated as a content SHA-256 after the local mismatch. No assumption was made that ETag and SHA-256 are interchangeable.

### Downloaded supporting artifacts

All were downloaded at the same pinned revision and hashed locally:

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 704 | `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843` |
| `generation_config.json` | 111 | `2056c988e990b0d13670f63f2f3b87b3b6d07edaf7a3416998ba27dab2d8a059` |
| `tokenizer.json` | 2,104,556 | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `tokenizer_config.json` | 3,658 | `4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424` |
| `special_tokens_map.json` | 831 | `e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3` |
| `merges.txt` | 466,391 | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| `vocab.json` | 800,662 | `82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79` |

No instruction-tuned or community model was acquired.

## Stop decision

Do not convert, quantize, or execute the model until the owner supplies an authoritative content SHA-256 or explicitly approves the documented mismatch with an independently verified provenance decision.
