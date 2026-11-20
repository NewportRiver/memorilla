from __future__ import annotations
import numpy as np

def iter_batches(store, collection_id=None, access_scope=None, chunk_size=128, contract=None, as_of=None):
    rows=[]
    for item in store.iter_vectors(collection_id, access_scope, chunk_size, contract, as_of):
        rows.append(item)
        if len(rows)==chunk_size:
            yield rows
            rows=[]
    if rows:
        yield rows

def vector_batches(store, collection_id=None, access_scope=None, chunk_size=128, contract=None):
    for rows in iter_batches(store, collection_id, access_scope, chunk_size, contract):
        yield np.stack([r['vector'] for r in rows]).astype(np.float32, copy=False), rows

def bounded_count(store, collection_id=None, access_scope=None, chunk_size=128, contract=None):
    return sum(len(rows) for rows in iter_batches(store, collection_id, access_scope, chunk_size, contract))
