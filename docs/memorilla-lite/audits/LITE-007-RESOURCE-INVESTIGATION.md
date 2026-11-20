# LITE-007B Resource Investigation

Status: complete for static and previously recorded evidence; no optimizer update performed.

## Runtime and restrictions

- Branch: `feature/memorilla-lite-android`
- Baseline: `5a48a6c22f3227ba82a0eef38c75e1725bf0372c`
- Python: native Termux Python 3.13.13
- PyTorch: 2.11.0
- Numerical threads: restricted to one for this investigation
- Computational subprocesses: one at a time
- Per-subprocess wall limit: 30 seconds
- RSS ceiling: 256 MiB
- Abort RAM threshold: 1,024 MiB; minimum operating RAM: 1,280 MiB
- No package installation, upgrade, model download, or optimizer update was performed

## Recorded guarded-pilot evidence

The existing LITE-007 training audit records:

- Android pilot configuration: batch size 1, maximum 8 updates, 30 seconds, two numerical threads in the historical attempt.
- Actual optimizer updates: 0.
- Stop reason: available RAM fell below the historical 768 MiB guard during guarded initialization.
- Peak recorded RSS: 1,581,723,648 bytes.
- No checkpoint was produced.
- Earlier standalone forward/backward smoke: 236,376,064 bytes RSS; it did not call an optimizer.

This is sufficient to establish that the approved 256 MiB RSS ceiling was exceeded. It does not identify a unique single allocation inside the historical process because the pilot did not persist phase-level telemetry.

## Operation boundary analysis

The current `lite/training/train.py` sequence is:

1. import NumPy, PyTorch, and training modules;
2. set PyTorch thread limits and random seeds;
3. load the complete development split and validation split, including precomputed embeddings;
4. construct `TorchAssociativeMemory`;
5. execute `gradient_audit`, which performs one forward, loss construction, and backward allocation;
6. construct `torch.optim.AdamW`;
7. enter the update loop, where `opt.zero_grad`, forward, backward, gradient clipping, and `opt.step` would occur.

The historical report only localizes the stop to guarded initialization; it does not prove whether the final trigger was dataset loading, model construction, gradient allocation, or optimizer construction. Repeating each phase independently would risk reproducing an operation already observed at 1.58 GB RSS and is prohibited by the current envelope.

## Static allocation analysis

The architecture has 429,312 parameters. At Float32:

| Component | Bytes | Approx. MiB |
|---|---:|---:|
| Parameters | 1,717,248 | 1.64 |
| One gradient tensor per parameter | 1,717,248 | 1.64 |
| AdamW first and second moments | 3,434,496 | 3.28 |
| Parameters + gradients + AdamW moments | 6,868,992 | 6.55 |
| One 137-record x 384-D memory tensor | 210,432 | 0.20 |

These numbers exclude PyTorch allocator arenas, autograd intermediates, temporary logits/attention tensors, Python objects, imported libraries, and model/runtime initialization. Therefore the raw model tensor footprint cannot explain the 1.58 GB RSS alone; native library and runtime overhead are dominant suspects, but are not proven as the unique trigger.

## Phase-level conclusions

| Phase | Evidence | Status |
|---|---|---|
| Python initialization | Current interpreter runs; no phase RSS trace | not isolated |
| NumPy initialization | NumPy 2.4.4 present; no phase RSS trace | not isolated |
| PyTorch initialization | PyTorch 2.11.0 imports; historical standalone smoke reached 236 MB | measured only in combined smoke |
| Model construction | static tensor footprint 1.64 MiB | not live-isolated |
| Parameter allocation | static tensor footprint 1.64 MiB | not live-isolated |
| Gradient allocation | historical backward smoke included; no phase trace | not isolated |
| Optimizer construction | current code constructs AdamW before loop; no step occurred | not isolated; no update |
| First optimizer step | never reached | not executed |
| Dataset loading | precomputed 1,500-record/1,200-query fixture exists; no phase trace | not isolated |

No optimizer update occurred. No new guarded training attempt was made.
