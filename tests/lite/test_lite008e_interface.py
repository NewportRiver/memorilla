from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "experiments/lite008e/embedding_harness.cpp"
PREPARE = ROOT / "experiments/lite008e/prepare_inputs.py"


def test_lite008e_harness_uses_pinned_external_embedding_contract():
    source = HARNESS.read_text(encoding="utf-8")
    for symbol in (
        "llama_model_n_embd_inp",
        "llama_batch_ext_add_token",
        "llama_batch_ext_add_embd",
        "llama_batch_ext_set_pos",
        "llama_batch_ext_set_output_logits",
        "llama_process",
        "llama_get_logits",
    ):
        assert symbol in source
    assert "LLAMA_PROCESS_TYPE_DECODE" in source
    assert "n_embd" in source


def test_lite008e_harness_rejects_malformed_inputs_before_submission():
    source = HARNESS.read_text(encoding="utf-8")
    assert "embedding rows are too short" in source
    assert "embedding contains non-finite value" in source
    assert "external index out of range" in source
    assert "context overflow before native submission" in source


def test_lite008e_input_preparation_is_isolated():
    source = PREPARE.read_text(encoding="utf-8")
    assert "local_files_only=True" in source
    assert "token_embd.weight" in source
    assert "out = Path(args.out)" in source
