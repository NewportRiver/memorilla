# LITE-008I Training-Resource Authorization Request

This request is informational and is not an authorization to execute.

The current 268,435,456-byte ceiling cannot support the official differentiable diagnostic because the local bfloat16 safetensors artifact alone is 269,060,552 bytes. A defensible training-only envelope must therefore exceed the serialized weight floor and include runtime, activations, logits, allocator fragmentation, and backward storage.

A future request should be based on a disposable subprocess and should specify:

- Maximum RSS in bytes
- Maximum wall duration
- One CPU thread and one subprocess
- Batch size 1
- Short sequence length
- One external memory embedding
- Frozen decoder parameters
- No decoder optimizer state
- Explicit timeout and process cleanup
- Forward and backward RSS separately
- Input-gradient norm and finite-difference checks
- No checkpoint publication and no production artifact mutation

No numeric envelope is proposed here because the full-model peak has not been measured and measuring it requires exceeding the currently approved ceiling. The smallest defensible proposal must be owner-approved before execution.
