"""Read-only identity gate; no numerical runtimes or optimizer operations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def identity_mismatches(source, claimed):
    fields = ('entity', 'query', 'answer_a', 'answer_b', 'partition')
    return {key: {'source': source.get(key), 'claimed': claimed.get(key)}
            for key in fields if source.get(key) != claimed.get(key)}


def verify(root=ROOT):
    def load(path):
        return json.loads(path.read_text())
    teachers = load(root / 'experiments/lite008u/teacher-dataset.json')['teachers']
    source_rows = {r['case_id']: r for r in load(root / 'experiments/lite008p/dataset-manifest.json')['rows']}
    claimed_rows = {r['case_id']: r for r in load(root / 'experiments/lite008q/dataset-manifest.json')['rows']}
    results = []
    for teacher in teachers:
        source_path = Path(teacher['source'])
        original = next(r for r in load(source_path)['results']
                        if (r['case_id'], r['target']) == (teacher['case_id'], teacher['target']))
        vector = teacher['embedding']
        digest = hashlib.sha256(json.dumps(vector, separators=(',', ':')).encode()).hexdigest()
        results.append({'case_id': teacher['case_id'], 'target': teacher['target'],
                        'dimensions': len(vector), 'recorded_hash_matches': digest == teacher['embedding_sha256'],
                        'source_vector_equal': vector == original['final_slots'][0],
                        'source_file_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        'identity_mismatches': identity_mismatches(source_rows[teacher['case_id']], claimed_rows[teacher['case_id']]),
                        'historical_margin': original['trajectory'][-1]['margin']})
    return {'schema': 'LITE-008V-provenance-v1', 'status': 'blocked_incompatible_identities',
            'rows': results, 'optimizer_updates': 0,
            'student_checkpoint_present': (root / 'experiments/lite008u/student-state.json').exists(),
            'decoder_reload_evaluation': 'not executed: stopped at incompatible provenance',
            'candidate_token_verification': 'not executed: answer-text identities already conflict',
            'encoder_feature_recovery': 'blocked: no unambiguous qualified training identity'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
