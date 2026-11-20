from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _marker(path, phase, **extra):
    row = {'phase': phase, 'timestamp_ns': time.time_ns(), **extra}
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def prepare_training_artifact(source, output, progress_path=None, partition='train', embedding_path=None, encoder_manifest_path=None):
    """Prepare only the verified train partition into a deterministic indexed artifact."""
    import numpy as np

    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(progress_path or output.with_suffix('.progress.jsonl'))
    dataset_path = source / 'dataset.json'
    embedding_path = Path(embedding_path or source / 'embeddings.npz')
    manifest_path = source / 'manifest.json'
    encoder_manifest_path = Path(encoder_manifest_path or source / 'embeddings.manifest.json')
    _marker(progress_path, 'source_open_start')
    data = json.loads(dataset_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    encoder_manifest = json.loads(encoder_manifest_path.read_text(encoding='utf-8'))
    if data.get('holdout_included') or manifest.get('holdout_labels_included'):
        raise ValueError('holdout material present in source artifact')
    _marker(progress_path, 'metadata_loaded', scenarios=len(data['scenarios']), records=len(data['records']), queries=len(data['queries']))
    with np.load(embedding_path, allow_pickle=False) as archive:
        record_ids = archive['record_ids'].tolist()
        query_ids = archive['query_ids'].tolist()
        record_embeddings = archive['record_embeddings']
        query_embeddings = archive['query_embeddings']
        record_index = {value: i for i, value in enumerate(record_ids)}
        query_index = {value: i for i, value in enumerate(query_ids)}
        records = {row['evidence_id']: row for row in data['records']}
        scenarios = {row['scenario_id']: row for row in data['scenarios']}
        selected_queries = [row for row in data['queries'] if scenarios[row['scenario_id']]['split'] == partition]
        max_records = max(len(scenarios[row['scenario_id']]['record_ids']) for row in selected_queries)
        n = len(selected_queries)
        q_vectors = np.empty((n, query_embeddings.shape[1]), dtype=np.float32)
        memories = np.zeros((n, max_records, record_embeddings.shape[1]), dtype=np.float32)
        valid = np.zeros((n, max_records), dtype=bool)
        positive = np.zeros((n, max_records), dtype=bool)
        qids = np.empty(n, dtype='U96')
        sids = np.empty(n, dtype='U64')
        eids = np.empty((n, max_records), dtype='U96')
        categories = np.empty(n, dtype='U64')
        for i, query in enumerate(selected_queries):
            scenario = scenarios[query['scenario_id']]
            eligible = []
            required = set(query['required_evidence_ids'])
            for evidence_id in scenario['record_ids']:
                record = records[evidence_id]
                if record['access_scope'] != query['access_scope']:
                    continue
                cutoff = query['historical_cutoff']
                if cutoff is not None and record['source_timestamp'] > cutoff:
                    continue
                eligible.append(evidence_id)
            qid = query['query_id']
            qids[i] = qid
            sids[i] = query['scenario_id']
            categories[i] = query['category']
            q_vectors[i] = query_embeddings[query_index[qid]]
            for j, evidence_id in enumerate(eligible):
                if evidence_id not in record_index:
                    raise ValueError(f'missing embedding for {evidence_id}')
                eids[i, j] = evidence_id
                memories[i, j] = record_embeddings[record_index[evidence_id]]
                valid[i, j] = True
                positive[i, j] = evidence_id in required
            if i == 0 or (i + 1) % 100 == 0:
                _marker(progress_path, 'examples_assembled', processed=i + 1, total=n)
    payload = {
        'query_embeddings': q_vectors,
        'memory_embeddings': memories,
        'valid_mask': valid,
        'positive_mask': positive,
        'query_ids': qids,
        'scenario_ids': sids,
        'evidence_ids': eids,
        'categories': categories,
    }
    temp = output.with_suffix('.npz.tmp')
    with temp.open('wb') as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output)
    metadata = {
        'schema_version': 1,
        'dataset_version': data['dataset_version'],
        'partition': partition,
        'query_count': n,
        'max_records': max_records,
        'source_manifest_sha256': _sha256(manifest_path),
        'source_dataset_sha256': _sha256(dataset_path),
        'source_embeddings_sha256': _sha256(embedding_path),
        'encoder_manifest_sha256': _sha256(encoder_manifest_path),
        'encoder': encoder_manifest['encoder'],
        'holdout_included': False,
        'content_sha256': _sha256(output),
    }
    meta_tmp = output.with_suffix('.json.tmp')
    meta_tmp.write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    os.replace(meta_tmp, output.with_suffix('.json'))
    _marker(progress_path, 'artifact_committed', query_count=n, content_sha256=metadata['content_sha256'])
    return metadata


class PreparedTrainingSet:
    def __init__(self, artifact, allow_validation=False):
        import numpy as np

        self.path = Path(artifact)
        self.metadata = json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8'))
        allowed = ('train', 'validation', 'development_evaluation') if allow_validation else ('train',)
        if self.metadata.get('holdout_included') or self.metadata.get('partition') not in allowed:
            raise ValueError('prepared artifact is not an authorized training partition')
        if _sha256(self.path) != self.metadata.get('content_sha256'):
            raise ValueError('prepared artifact checksum mismatch')
        self.archive = np.load(self.path, allow_pickle=False)
        self.length = int(self.metadata['query_count'])
        if self.archive['query_embeddings'].shape[0] != self.length:
            raise ValueError('prepared artifact query count mismatch')

    def __len__(self):
        return self.length

    def eligible_indices(self):
        positive = self.archive['positive_mask']
        valid = self.archive['valid_mask']
        return [i for i in range(self.length)
                if bool(positive[i].any()) and int(valid[i].sum()) > int(positive[i].sum())]

    def get(self, index):
        import numpy as np

        if not 0 <= index < self.length:
            raise IndexError(index)
        return {
            'query_embedding': np.asarray(self.archive['query_embeddings'][index], dtype=np.float32),
            'memory_embeddings': np.asarray(self.archive['memory_embeddings'][index], dtype=np.float32),
            'valid_mask': np.asarray(self.archive['valid_mask'][index], dtype=bool),
            'positive_mask': np.asarray(self.archive['positive_mask'][index], dtype=bool),
            'query_id': str(self.archive['query_ids'][index]),
            'scenario_id': str(self.archive['scenario_ids'][index]),
            'evidence_ids': [str(x) for x in self.archive['evidence_ids'][index] if x],
            'category': str(self.archive['categories'][index]),
        }

    def close(self):
        self.archive.close()