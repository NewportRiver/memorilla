import copy
import json
from pathlib import Path
import pytest
from experiments.lite008vr1.identity import digest, file_hash, seal, validate

ROOT = Path(__file__).resolve().parents[2]


def records():
    bindings = json.loads((ROOT/'docs/memorilla-lite/audits/LITE-008V-R1-PROVENANCE-MANIFEST.json').read_text())['records']
    vectors = json.loads((ROOT/'experiments/lite008u/teacher-dataset.json').read_text())['teachers']
    return [(dict(identity=b['identity'], identity_sha256=b['identity_sha256']), t['embedding'])
            for b, t in zip(bindings, vectors)]


def test_all_original_bindings_reload():
    for binding, vector in records():
        assert validate(binding, binding, vector)


@pytest.mark.parametrize('field', ['manifest_sha256', 'entity', 'evidence', 'query',
                                  'candidate_token_ids', 'direction', 'partition',
                                  'vector_sha256', 'model', 'generation', 'candidate_answers'])
def test_fail_closed_for_each_identity_dimension(field):
    binding, vector = records()[0]
    supplied = copy.deepcopy(binding)
    supplied['identity'][field] = 'tampered'
    supplied['identity_sha256'] = digest(supplied['identity'])
    with pytest.raises(ValueError, match='FAIL_CLOSED'):
        validate(binding, supplied, vector)


def test_eight_exact_historical_p_q_collisions_rejected():
    qpath = ROOT/'experiments/lite008q/dataset-manifest.json'
    q = {r['case_id']: r for r in json.loads(qpath.read_text())['rows']}
    count = 0
    for binding, vector in records():
        supplied = copy.deepcopy(binding)
        identity = supplied['identity']
        row = q[identity['case_id']]
        identity.update(manifest_sha256=file_hash(qpath), partition=row['partition'],
                        entity=row['entity'], query=row['query'],
                        candidate_answers=[row['answer_a'], row['answer_b']],
                        evidence=row['evidence_a' if identity['direction']=='A' else 'evidence_b'])
        supplied['identity_sha256'] = digest(identity)
        with pytest.raises(ValueError, match='FAIL_CLOSED'):
            validate(binding, supplied, vector)
        count += 1
    assert count == 8


def test_vector_corruption_rejected():
    binding, vector = records()[0]
    vector[0] += 1
    with pytest.raises(ValueError, match='vector'):
        validate(binding, binding, vector)


def test_missing_identity_field_rejected():
    binding, _ = records()[0]
    del binding['identity']['evidence']
    with pytest.raises(ValueError, match='incomplete'):
        seal(binding['identity'])
