import json
from pathlib import Path

AUDIT=Path(__file__).parents[2]/'docs/memorilla-lite/audits'

def load(name):
    return json.loads((AUDIT/name).read_text())

def test_lite008o_reports_are_valid_and_complete():
    names=list(AUDIT.glob('LITE-008O-*.json'))
    assert len(names)==9
    for path in names:
        assert load(path.name)['schema'].startswith('LITE-008O-')

def test_lite008o_dataset_is_partitioned_and_single_token():
    data=load('LITE-008O-DATASET-MANIFEST.json')
    assert len(data['calibration_case_ids'])==8
    assert len(data['oracle_case_ids'])==8
    assert data['partitions_entity_disjoint'] is True
    assert all(v==1 for v in data['primary_candidate_token_lengths'].values())

def test_lite008o_primary_oracle_no_go_is_preserved():
    data=load('LITE-008O-COUNTERFACTUAL-ORACLE-RESULTS.json')
    assert data['required_successes']==4
    assert data['successful_pairs']==0
    assert data['status']=='failed_primary_gate'

def test_lite008o_forward_only_has_no_backward_vjps():
    data=load('LITE-008O-FORWARD-ONLY-QUALIFICATION.json')
    assert data['textual_backward_vjps']==0
    assert data['final_embedding_backward_vjps']==0
    assert data['separate_paths'] is True

def test_lite008o_update_budget_and_controls():
    telemetry=load('LITE-008O-RESOURCE-TELEMETRY.json')
    assert telemetry['total_optimizer_updates']==96
    assert telemetry['total_optimizer_updates']<=256
    controls=load('LITE-008O-CONTROL-COMPARISON.json')
    assert set(controls['directions'])=={'zero','scale_matched_random','genuine_answer_a','genuine_answer_b','initial_eos','random_search'}
    assert controls['random_search_candidates_per_case']==8

def test_lite008o_gated_branches_are_not_claimed():
    assert load('LITE-008O-BF16-COMPARISON.json')['status']=='unexecuted_gated'
    assert load('LITE-008O-QUANTIZED-TRANSFER-RESULTS.json')['status']=='unexecuted_gated'
