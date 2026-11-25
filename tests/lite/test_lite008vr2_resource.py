import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / 'docs/memorilla-lite/audits'


def test_resource_classes_are_distinct_and_production_unchanged():
    text = (AUD / 'LITE-008V-R2-RESOURCE-CLASSIFICATION.md').read_text()
    assert '268,435,456' in text
    assert '1,073,741,824' in text
    assert 'not production inference' in text


def test_reference_telemetry_has_zero_updates():
    data = json.loads((AUD / 'LITE-008V-R2-RESOURCE-TELEMETRY.json').read_text())
    assert data['classification'] == 'scientific_reference_evaluation'
    assert data['optimizer_updates'] == 0
    assert data['production_inference_ceiling_unchanged'] == 268435456


def test_training_remains_unexecuted():
    text = (AUD / 'LITE-008V-R2-RESOURCE-CLASSIFICATION.md').read_text()
    assert 'no training was performed' in text
