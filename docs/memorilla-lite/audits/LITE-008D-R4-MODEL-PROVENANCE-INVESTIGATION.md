# LITE-008D-R4 Model Provenance Investigation

## Resolution

The model integrity gate passes. The previous discrepancy was an identifier-classification error:

- `8595fbb3...` is the Xet file hash and was returned as the HTTP `ETag`.
- `80521b4028...` is the Git LFS content SHA-256 and matches the locally calculated SHA-256.
- `ed736e4e...` is the Git blob identifier and is not a SHA-256 of the downloaded file.

The acquired artifact was retained and was not re-downloaded.

## Exact repository identity

- Repository: `HuggingFaceTB/SmolLM2-135M`
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- File: `model.safetensors`
- Revision API: `https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M/revision/93efa2f097d58c2a74874c7e644dbc9b0cee75a2?blobs=true`
- Tree API: `https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2?expand=true`
- Raw evidence directory: `$HOME/.cache/memorilla-lite/LITE-008D-R4/provenance/`
- Retrieval timestamp: `2026-11-24T10:25:01Z` UTC (headers received at approximately `10:25:03Z`)

## Authoritative metadata

The revision API reported:

- `blobId`: `ed736e4e5f39dd99c5d435bf84c0d8c222839b1b`
- `size`: `269060552`
- `lfs.sha256`: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- `lfs.size`: `269060552`
- `lfs.pointerSize`: `134`

The expanded tree API independently reported:

- `oid`: `ed736e4e5f39dd99c5d435bf84c0d8c222839b1b`
- `size`: `269060552`
- `lfs.oid`: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- `lfs.size`: `269060552`
- `xetHash`: `8595fbb3d6026b3818e1a9d538e145d63abbdb21e948de9b4968b5a16e96a579`
- `path`: `model.safetensors`

The download response headers reported:

- `ETag`: `"8595fbb3d6026b3818e1a9d538e145d63abbdb21e948de9b4968b5a16e96a579"`
- `Content-Length`: `269060552`
- `Content-Disposition`: `model.safetensors`

## Local verification

- Local path: `$HOME/.cache/memorilla-lite/LITE-008D-R3/model/model.safetensors`
- Local size: `269060552` bytes
- Local SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- Local SHA-256 equals authoritative `lfs.sha256`: yes
- Local size equals authoritative LFS size: yes

Configuration and tokenizer files were acquired from the same pinned revision and retained with their local hashes in the R3 acquisition evidence. No community or instruction-tuned artifact was used.

## Identifier classification

- Git blob ID: repository object identifier for the file; not the downloaded content SHA-256.
- HTTP ETag: cache/storage validator; here equal to the Xet hash; not the content SHA-256.
- Git LFS SHA-256: authoritative content hash for the LFS payload; matches the local hash.
- Xet file hash: storage-layer identifier; here exposed as ETag; not interchangeable with LFS SHA-256.
- Local SHA-256: independently calculated payload hash; accepted only because it matches authoritative LFS metadata.

The original R3 manifest is preserved as historical evidence. The corrected R4 manifest labels the identifiers according to their verified meanings.
