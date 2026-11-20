import hashlib,sqlite3,subprocess,sys,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from lite.runtime.preflight import ResourceLimits,preflight
from lite.runtime.smoke_test import run
from lite.runtime.telemetry import rss_bytes,thread_count

def test_preflight_and_smoke():
    x=run(); assert x['preflight']['ok']; assert x['sqlite_fts5']; assert x['matrix_shape']==[32,32]; assert x['sha256']==hashlib.sha256(b'\x00synthetic evidence').hexdigest()

def test_numpy_numeric_types_and_finite():
    a=np.arange(16,dtype=np.float32).reshape(4,4); assert np.isfinite(a.astype(np.float16)).all(); q=np.clip(a,-127,127).astype(np.int8); assert q.dtype==np.int8

def test_sqlite_blob_and_fts5():
    c=sqlite3.connect(':memory:'); c.execute('create table x(b blob)'); c.execute('create virtual table f using fts5(t)'); b=b'abc\x00'; c.execute('insert into x values(?)',(b,)); c.execute('insert into f values(?)',('alpha beta',)); c.commit(); assert c.execute('select b from x').fetchone()[0]==b; assert c.execute("select count(*) from f where f match 'alpha'").fetchone()[0]==1

def test_timeout_cleanup():
    p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);
    try:
        try: p.wait(timeout=0.2)
        except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=2)
        assert p.poll() is not None
    finally:
        if p.poll() is None: p.kill(); p.wait()

def test_telemetry_available_or_explicitly_unavailable():
    assert rss_bytes() is None or rss_bytes()>0; assert thread_count() is None or thread_count()>0
