"""Synthetic bounded smoke tests for the installed Python/NumPy/SQLite runtime."""
from __future__ import annotations
import argparse,hashlib,sqlite3,tempfile,time
from pathlib import Path
import numpy as np
from .preflight import ResourceLimits,preflight
from .telemetry import snapshot

def run():
    started=time.monotonic_ns(); limits=ResourceLimits(); pre=preflight('/tmp',limits,0)
    if not pre['ok']: raise RuntimeError(pre['reasons'])
    rng=np.random.default_rng(20261119); a=rng.standard_normal((32,32),dtype=np.float32); b=rng.standard_normal((32,32),dtype=np.float32)
    f32=(a@b).astype(np.float32); f16=(a.astype(np.float16)@b.astype(np.float16)).astype(np.float16)
    q=np.clip(np.rint(a*32),-127,127).astype(np.int8); deq=q.astype(np.float32)/32.0; qerr=float(np.max(np.abs(a-deq)))
    with tempfile.TemporaryDirectory(prefix='memorilla-lite-smoke-') as td:
        db=sqlite3.connect(Path(td)/'smoke.db'); db.execute('create table records(id integer primary key, body blob, digest text)'); db.execute('create virtual table search using fts5(body)')
        blob=bytes([0])+b'synthetic evidence'; digest=hashlib.sha256(blob).hexdigest(); db.execute('insert into records(body,digest) values(?,?)',(blob,digest)); db.execute('insert into search(body) values(?)',('synthetic evidence',)); db.commit()
        assert db.execute('select body from records where digest=?',(digest,)).fetchone()[0]==blob
        assert db.execute("select count(*) from search where search match 'synthetic'").fetchone()[0]==1
        db.close()
    assert np.isfinite(f32).all() and np.isfinite(f16).all()
    return {'preflight':pre,'matrix_shape':list(f32.shape),'float16_max_abs_error':float(np.max(np.abs(f32-f16.astype(np.float32)))),'int8_dequant_max_abs_error':qerr,'telemetry':snapshot('smoke',started,'/tmp').as_dict(),'sqlite_fts5':True,'sha256':digest}

if __name__=='__main__':
    import json; print(json.dumps(run(),sort_keys=True))
