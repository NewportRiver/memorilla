from experiments.lite008v.verify_provenance import identity_mismatches, verify


def test_case_id_does_not_establish_identity():
    a = dict(case_id='case_00', entity='one', query='q1', answer_a='red', answer_b='blue', partition='calibration')
    b = dict(a, entity='two', query='q2', answer_a='green', partition='train')
    assert set(identity_mismatches(a, b)) == {'entity', 'query', 'answer_a', 'partition'}


def test_identical_identity():
    assert identity_mismatches({'query': 'same'}, {'query': 'same'}) == {}


def test_retained_teacher_identity_conflict():
    report = verify()
    assert len(report['rows']) == 8
    assert all(row['recorded_hash_matches'] and row['source_vector_equal'] for row in report['rows'])
    assert all('query' in row['identity_mismatches'] for row in report['rows'])
    assert report['optimizer_updates'] == 0
