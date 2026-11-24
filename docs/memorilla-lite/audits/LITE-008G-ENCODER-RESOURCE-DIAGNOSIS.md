# LITE-008G Encoder Resource Diagnosis

The prior 892,514,304-byte observation came from a combined process that initialized MiniLM, imported the projection/evaluation stack, loaded GGUF tooling, and dequantized the decoder embedding matrix. The qualified MiniLM path was measured phase-by-phase in isolated Python subprocesses without PyTorch.

| Phase | Peak RSS | Peak PSS | Result |
|---|---:|---:|---|
| Python startup | 9,875,456 | 5,752,832 | success |
| NumPy import | 26,492,928 | 21,491,712 | success |
| tokenizer initialization | 35,446,784 | 29,827,072 | success |
| ONNX Runtime import | 41,365,504 | 36,320,256 | success |
| ONNX session initialization | 77,484,032 | 72,547,328 | success |
| single-record encoding | 86,601,728 | 80,041,984 | success |
| 24-record batch encoding | 92,258,304 | 86,238,208 | success |
| feature serialization | 91,725,824 | 85,733,376 | success |

The existing implementation uses one ONNX Runtime session, one numerical thread, and no PyTorch import. The batch and serialization steps add only a small amount over session plus single-record encoding. No duplicate MiniLM model load or unnecessary runtime replacement was found. ONNX Runtime emitted its existing Android support warning; no dependency was installed.

Cached feature matrix SHA-256: `99877dd1cbeacdd82bb74ee21753644e216b3f1df2968c71d06b2d8174f8143a`.

The encoder is substantially below the decoder-specific 268,435,456-byte ceiling in isolation. This does not establish whole-pipeline or thermal qualification. Raw phase telemetry was preserved at `~/.cache/memorilla-lite/LITE-008G/encoder-resource-diagnosis.json`.
