from experiments.lite008i.gradient_diagnostic import run


def test_synthetic_frozen_decoder_input_gradient_path():
    result = run()
    assert result["decoder_parameters_frozen"]
    assert result["decoder_parameter_grads_absent"]
    assert result["external_embedding_gradient_finite"]
    assert result["relative_error"] < 1e-3
    assert result["embedding_optimization_reduced_loss"]
