from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np


def _text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True))
    os.replace(tmp, path)


def prepare_incremental(dataset_path, output_path, encoder, progress_path=None, fail_after=None, batch_size=32):
    dataset_path = Path(dataset_path); output_path = Path(output_path)
    progress_path = Path(progress_path or output_path.with_suffix('.progress.json'))
    output_path.parent.mkdir(parents=True, exist_ok=True); progress_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = progress_path.with_suffix('.cache.npz')
    data = json.loads(dataset_path.read_text())
    items = [('record', x['evidence_id'], x['content']) for x in data['records']] + [('query', x['query_id'], x['query']) for x in data['queries']]
    source_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    state = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if state and state.get('source_hash') != source_hash:
        raise ValueError('source identity mismatch')
    hashes = []; vectors = []
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        hashes = cache['hashes'].astype(str).tolist(); vectors = [x for x in cache['vectors']]
    known = {h: v for h, v in zip(hashes, vectors)}
    next_index = int(state.get('next_index', 0)); encoded = 0; reused = 0
    for start in range(next_index, len(items), batch_size):
        batch = items[start:start + batch_size]; missing = []; missing_hashes = []
        for _, _, text in batch:
            h = _text_hash(text)
            if h in known: reused += 1
            else: missing.append(text); missing_hashes.append(h)
        if missing:
            values = np.asarray(encoder.encode_batch(missing), dtype=np.float32)
            if values.ndim != 2 or not np.isfinite(values).all(): raise ValueError('invalid embedding output')
            for h, value in zip(missing_hashes, values): known[h] = value; encoded += 1
        ordered_hashes = list(known); ordered_vectors = np.asarray([known[h] for h in ordered_hashes], dtype=np.float32)
        tmp_cache = cache_path.with_suffix('.tmp.npz')
        with tmp_cache.open('wb') as handle:
            np.savez_compressed(handle, hashes=np.asarray(ordered_hashes, dtype='U64'), vectors=ordered_vectors); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp_cache, cache_path)
        next_index = start + len(batch)
        _atomic_json(progress_path, {'schema_version': 2, 'source_hash': source_hash, 'next_index': next_index, 'encoded': encoded, 'reused': reused, 'cache_count': len(known)})
        if fail_after is not None and next_index >= fail_after: raise RuntimeError('fault injection after embedding progress commit')
    records = np.asarray([known[_text_hash(x['content'])] for x in data['records']], np.float32)
    queries = np.asarray([known[_text_hash(x['query'])] for x in data['queries']], np.float32)
    tmp = output_path.with_suffix('.tmp.npz')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, record_ids=np.asarray([x['evidence_id'] for x in data['records']], dtype='U128'), query_ids=np.asarray([x['query_id'] for x in data['queries']], dtype='U128'), record_embeddings=records, query_embeddings=queries)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, output_path)
    meta = {'schema_version': 2, 'dataset_version': data['dataset_version'], 'source_hash': source_hash, 'record_count': len(records), 'query_count': len(queries), 'dimension': int(records.shape[1]), 'unique_texts': len(known), 'encoded_texts': encoded, 'reused_texts': reused, 'artifact_sha256': hashlib.sha256(output_path.read_bytes()).hexdigest()}
    _atomic_json(output_path.with_suffix('.json'), meta)
    return meta
