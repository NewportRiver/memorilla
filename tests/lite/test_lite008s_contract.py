import json
from pathlib import Path

def test_s_qualification_artifact_is_measured_and_zero_update():
 p=Path(__file__).parents[2]/'experiments/lite008s/gradient-qualification-s.json'; q=json.loads(p.read_text())
 assert q['schema']=='LITE-008S-gradient-qualification-v1'
 assert q['status']=='qualified'
 assert q['optimizer_updates']==0
 assert all(q['stages'][s]['status']=='pass' for s in ('stage_A','stage_B','stage_C','stage_D'))
 assert q['numerical_configuration']['forward_only_finite_difference'] is True

def test_s_required_reports_exist_and_parse():
 p=Path(__file__).parents[2]/'docs/memorilla-lite/audits'
 names=['LITE-008S-MINIMAL-REPRODUCTION.json','LITE-008S-DETERMINISM-RESULTS.json','LITE-008S-INPUT-ISOLATION-RESULTS.json','LITE-008S-LAYER-BOUNDARY-COMPARISON.json','LITE-008S-COMPONENT-VJP-RESULTS.json','LITE-008S-REDUCED-QUALIFICATION.json','LITE-008S-FULL-DECODER-QUALIFICATION.json','LITE-008S-FAIL-CLOSED-GATE-VERIFICATION.json','LITE-008S-RESOURCE-TELEMETRY.json']
 for n in names: assert json.loads((p/n).read_text())
