"""Recover provenance only. Never imports a decoder or executes optimization."""
import json
import shutil
import struct
import time
from pathlib import Path
from experiments.lite008vr1.identity import digest, file_hash, seal, validate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/memorilla-lite/audits'
CACHE = Path.home() / '.cache/memorilla-lite/LITE-008V-R1'
MODEL = Path.home() / '.cache/memorilla-lite/LITE-008D-R3/model'


def run():
    from tokenizers import Tokenizer
    started = time.monotonic()
    CACHE.mkdir(exist_ok=True, parents=True)
    source = Path.home() / '.cache/memorilla-lite/LITE-008P/lite008u-teachers.json'
    manifest = ROOT / 'experiments/lite008p/dataset-manifest.json'
    data = json.loads(source.read_text())
    rows = {r['case_id']: r for r in json.loads(manifest.read_text())['rows']}
    model = {name: file_hash(MODEL / name) for name in
             ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')}
    reference = json.loads((ROOT / 'experiments/lite008t/contrastive-qualification.json').read_text())['decoder_hashes']
    assert all(model[k] == v for k, v in reference.items())
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    records = []
    artifacts = []
    for original, name in ((source, 'original-generation.json'), (manifest, 'original-p-manifest.json')):
        destination = CACHE / name
        if destination.exists() and file_hash(destination) != file_hash(original):
            raise ValueError('FAIL_CLOSED: retained artifact collision')
        if not destination.exists():
            shutil.copyfile(original, destination)
        assert file_hash(destination) == file_hash(original)
        artifacts.append({'path': str(destination), 'bytes': destination.stat().st_size,
                          'sha256': file_hash(destination), 'reload_verified': True})
    for result in data['results']:
        row = rows[result['case_id']]
        vector = result['final_slots'][0]
        candidate_ids = [tokenizer.encode(' ' + row[k], add_special_tokens=False).ids
                         for k in ('answer_a', 'answer_b')]
        assert all(len(ids) == 1 for ids in candidate_ids)
        identity = {'manifest_sha256': file_hash(manifest), 'partition': row['partition'],
                    'entity': row['entity'], 'case_id': row['case_id'],
                    'evidence': row['relevant_evidence' if result['target'] == 'A' else 'counterfactual_evidence'],
                    'query': row['query'], 'candidate_answers': [row['answer_a'], row['answer_b']],
                    'candidate_token_ids': candidate_ids, 'direction': result['target'],
                    'model': model, 'generation': {k: result[k] for k in ('slots', 'init', 'objective_kind', 'lr', 'updates')},
                    'vector_sha256': digest(vector), 'purpose': 'historical_reconstruction_diagnostic'}
        binding = seal(identity)
        validate(binding, binding, vector)
        records.append({**binding, 'source_result_key': [result['case_id'], result['target']],
                        'source_file_sha256': file_hash(source), 'vector_dimensions': len(vector),
                        'historical_final_margin': result['trajectory'][-1]['margin'],
                        'evidence_note': 'Associated P factual evidence; isolated generation consumed query and target, not evidence embeddings.'})
    assert len(records) == 8
    report = {'status': 'corrected_provenance_recovered_not_functionally_requalified',
              'source_manifest': str(manifest), 'source_manifest_sha256': file_hash(manifest),
              'generation_implementation_sha256': file_hash(ROOT / 'experiments/lite008p/p_experiment.py'),
              'records': records, 'retained_artifacts': artifacts,
              'identity_limitations': 'Historical generation output did not embed manifest/model hashes; recovery combines unchanged repository source, recorded invocation and current content checks against T. Not cryptographic proof of historical runtime custody.',
              'candidate_tokenization': 'Frozen tokenizer.json backend; original single-token space-prefixed answer encoding, no special tokens.'}
    with (MODEL / 'model.safetensors').open('rb') as stream:
        header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
    payload = sum(x['data_offsets'][1] - x['data_offsets'][0] for k, x in header.items() if k != '__metadata__')
    report['decoder_resident_tensor_payload_bytes'] = payload
    report['inference_rss_ceiling_bytes'] = 1073741824
    report['original_decoder_path_preflight'] = 'authorized_scientific_reference_evaluation'
    output = OUT / 'LITE-008V-R1-PROVENANCE-MANIFEST.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'records': len(records), 'tensor_payload_bytes': payload,
                      'preflight': report['original_decoder_path_preflight'], 'seconds': time.monotonic()-started}))


if __name__ == '__main__':
    run()
