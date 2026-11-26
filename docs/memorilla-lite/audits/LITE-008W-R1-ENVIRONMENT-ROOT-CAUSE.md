# LITE-008W-R1 Environment Root Cause

Root cause: W was invoked with system Termux Python at `/data/data/com.termux/files/usr/bin/python` and its system site-packages. That interpreter had torch 2.11.0, NumPy 2.4.4 and ONNX Runtime 1.27.0, but not transformers, tokenizers, or datasets.

The previously qualified LITE-008D-R4 environment was present at `~/.cache/memorilla-lite/LITE-008D-R4/venv`. Its explicit interpreter path resolves through the Termux Python binary but has a distinct venv prefix and venv site-packages. It contains transformers 4.57.6, tokenizers 0.22.2 and safetensors 0.8.0.

This was an interpreter-selection failure, not evidence of decoder or architecture insufficiency. The venv uses `include-system-site-packages=true`, so torch/NumPy/ONNX Runtime are inherited from the qualified Termux installation while decoder packages are supplied by the venv.

ONNX Runtime emits an Android unsupported-platform warning but imports and the retained R2 runtime qualification completed successfully.
