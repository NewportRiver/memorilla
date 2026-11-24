import json
from pathlib import Path

AUDIT=Path(__file__).parents[2]/'docs/memorilla-lite/audits'

def load(name):
    return json.loads((AUDIT/name).read_text())

def test_lite008p_json_reports_parse():
    paths=list(AUDIT.glob('LITE-008P-*.json'))
    assert len(paths)==10
    for path in paths:
        assert load(path.name)['schema'].startswith('LITE-008P-')

def test_lite008p_fresh_oracle_passes_four_of_four():
    data=load('LITE-008P-INDEPENDENT-ORACLE-RESULTS.json')
    assert data['required_successes']==4
    assert data['successful_counterfactual_pairs']==4
    assert all(row['pair_pass'] for row in data['pairs'])

def test_lite008p_objective_components_are_measured():
    data=load('LITE-008P-OBJECTIVE-GRADIENT-ANALYSIS.json')['diagnostic']['result']
    assert data['nll_margin_cosine'] < 0.9
    assert data['nll']['gradient_norm'] > 0
    assert data['margin']['gradient_norm'] > 0

def test_lite008p_multislot_gradient_check():
    data=load('LITE-008P-MULTISLOT-NUMERICAL-QUALIFICATION.json')['representative']
    assert data['finite_difference_relative_error'] < 0.01
    assert all(x > 0 for x in data['multislot_gradient_norms'])

def test_lite008p_budget_and_gates():
    data=load('LITE-008P-RESOURCE-TELEMETRY.json')
    assert data['optimizer_updates_completed'] <= 256
    assert data['resource_violations']==[]
    assert load('LITE-008P-QUANTIZED-TRANSFER-RESULTS.json')['status']=='completed_separate_diagnostic'

def test_lite008p_dataset_is_fresh_and_disjoint():
    data=load('LITE-008P-DATASET-MANIFEST.json')
    assert data['entity_disjoint'] is True
    assert data['calibration_case_ids']==[f'case_{i:02d}' for i in range(8)]
    assert data['oracle_case_ids']==[f'case_{i:02d}' for i in range(8,16)]

def test_lite008p_scope_remains_isolated():
    decision=(AUDIT/'LITE-008P-GO-NOGO-DECISION.md').read_text()
    assert 'Generalized semantic bridge training' in decision
    assert 'prohibited' in decision
