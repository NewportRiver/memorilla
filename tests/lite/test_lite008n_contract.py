import json
from pathlib import Path

AUDIT=Path(__file__).parents[2]/'docs/memorilla-lite/audits'

def load(name):
    return json.loads((AUDIT/name).read_text())

def test_lite008n_required_reports_are_valid_json():
    names=[p.name for p in AUDIT.glob('LITE-008N-*.json')]
    assert len(names) == 6
    for name in names:
        assert load(name)['schema'].startswith('LITE-008N-')

def test_lite008n_reduced_equivalence_passed():
    data=load('LITE-008N-REFERENCE-EQUIVALENCE-RESULTS.json')
    assert data['all_reduced_tolerances_passed'] is True
    assert data['external_gradient_max_absolute_error'] == 0.0

def test_lite008n_full_reference_includes_all_layers():
    data=load('LITE-008N-FULL-MODEL-REFERENCE-RESULTS.json')
    assert data['status']=='qualified'
    assert data['layers']==30
    assert data['decoder_parameters_frozen'] is True

def test_lite008n_preserves_bf16_partial_failure():
    data=load('LITE-008N-BF16-GRADIENT-COMPARISON.json')
    assert data['status']=='partial_threshold_failure'
    assert data['threshold_results']['coordinate_sign_agreement'] is False

def test_lite008n_usefulness_has_no_update_control():
    data=load('LITE-008N-GRADIENT-USEFULNESS-RESULTS.json')
    assert data['no_update_control'] is True
    assert len(data['no_update_nlls']) == 3
