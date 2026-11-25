import json
from pathlib import Path
AUD=Path(__file__).parents[2]/'docs/memorilla-lite/audits'
def load(n): return json.loads((AUD/n).read_text())
def test_q_json_reports_parse_and_required_scope():
 names=['LITE-008Q-CORRECTED-TRANSFER-RESULTS.json','LITE-008Q-GRADIENT-CHAIN-QUALIFICATION.json','LITE-008Q-DATASET-MANIFEST.json','LITE-008Q-BRIDGE-TRAINING-RESULTS.json','LITE-008Q-INDEPENDENT-DEVELOPMENT-RESULTS.json','LITE-008Q-CONTROL-COMPARISON.json','LITE-008Q-CHECKPOINT-INTEGRITY.json','LITE-008Q-RESOURCE-TELEMETRY.json']
 assert all((AUD/n).is_file() for n in names)
 for n in names: load(n)
def test_q_dataset_is_entity_disjoint_and_partitioned():
 d=load('LITE-008Q-DATASET-MANIFEST.json')['manifest'];rows=d['rows'];assert len(rows)==20;assert len(d['train_case_ids'])==8;assert len(d['calibration_case_ids'])==4;assert len(d['development_case_ids'])==8;assert len({r['entity'] for r in rows})==20
def test_q_chain_is_explicitly_recorded():
 d=load('LITE-008Q-GRADIENT-CHAIN-QUALIFICATION.json');assert d['finite_bridge_gradients'];assert d['encoder_parameters_frozen'];assert d['decoder_parameters_frozen'];assert d['max_relative_error']>0.10
def test_q_development_no_go_is_not_silently_promoted():
 d=load('LITE-008Q-INDEPENDENT-DEVELOPMENT-RESULTS.json');assert d['summary']['reversal_rate']=='3/8'
def test_q4_corrected_scoring_is_independent():
 d=load('LITE-008Q-CORRECTED-TRANSFER-RESULTS.json');assert 'independent single-token' in d['scoring'];assert d['reversal_rate']=='2/4'
