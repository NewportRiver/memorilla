import json
from pathlib import Path

AUDIT=Path(__file__).parents[2]/'docs/memorilla-lite/audits'

def test_lite008m_required_reports_are_valid_json():
    names=['LITE-008M-BALANCED-DATASET-MANIFEST.json','LITE-008M-LOADING-RESOURCE-RESULTS.json','LITE-008M-FP32-REFERENCE-RESULTS.json','LITE-008M-COMPONENT-REFERENCE-RESULTS.json','LITE-008M-BF16-GRADIENT-COMPARISON.json','LITE-008M-GRADIENT-USEFULNESS-RESULTS.json','LITE-008M-TEXTUAL-COMPETENCE-RESULTS.json','LITE-008M-RESOURCE-TELEMETRY.json']
    for name in names:
        data=json.loads((AUDIT/name).read_text())
        assert data['schema'].startswith('LITE-008M-')

def test_lite008m_balanced_manifest_contract():
    data=json.loads((AUDIT/'LITE-008M-BALANCED-DATASET-MANIFEST.json').read_text())
    assert data['primary_single_token_pairs'] >= 16
    assert data['correct_answer_orientation']=={'A':12,'B':12}
    assert data['candidate_order_counterbalanced'] is True
    assert data['entity_overlap_between_partitions'] is False

def test_lite008m_full_reference_is_not_overclaimed():
    data=json.loads((AUDIT/'LITE-008M-FP32-REFERENCE-RESULTS.json').read_text())
    assert data['status']=='blocked_by_rss_ceiling'
    component=json.loads((AUDIT/'LITE-008M-COMPONENT-REFERENCE-RESULTS.json').read_text())
    assert component['status']=='completed_component_only'
    assert component['full_transformer_layers_included'] is False
