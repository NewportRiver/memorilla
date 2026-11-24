"""Prepare isolated LITE-008E token IDs and genuine Q4 input embeddings.

This script intentionally writes only under the caller-selected cache directory.
It reads tokenizer.json through the isolated R4 environment and dequantizes only
 token_embd.weight from the already-qualified Q4 GGUF for interface equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, dequantize
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--q4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="The capital of France is")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    ids = tokenizer(args.text, add_special_tokens=True)["input_ids"]
    reader = GGUFReader(args.q4)
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    matrix = dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)
    if matrix.ndim != 2 or matrix.shape[1] != 576:
        raise ValueError(f"unexpected embedding matrix shape: {matrix.shape}")
    rows = matrix[np.asarray(ids, dtype=np.int64)]
    np.asarray(ids, dtype="<i4").tofile(out / "ids.bin")
    rows.astype("<f4").tofile(out / "embeddings-q4.f32")
    (out / "input.txt").write_text(args.text + "\n", encoding="utf-8")
    print({
        "ids": ids,
        "tokens": tokenizer.convert_ids_to_tokens(ids),
        "shape": list(rows.shape),
        "ids_sha256": sha256(out / "ids.bin"),
        "embeddings_sha256": sha256(out / "embeddings-q4.f32"),
    })


if __name__ == "__main__":
    main()
