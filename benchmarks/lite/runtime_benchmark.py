"""Bounded synthetic benchmark; no model or persistent fixture output."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from lite.attention.reference import attention,deterministic_weights
from lite.runtime.preflight import ResourceLimits,preflight
from lite.runtime.telemetry import snapshot

def run(sizes=(100,1000,5000),dim=384,slots=8,chunk=512,reps=2):
    pre=preflight('/tmp',ResourceLimits(),0); rows=[]; w=deterministic_weights(dim,slots)
    if not pre['ok']: return {'status':'blocked','preflight':pre,'results':[]}
    for n in sizes:
        rng=np.random.default_rng(1000+n); mem=rng.standard_normal((n,dim),dtype=np.float32); q=rng.standard_normal(dim,dtype=np.float32); times=[]; maxerr=None; start=time.monotonic_ns(); before=snapshot(f'before-{n}',start,'/tmp')
        for _ in range(reps):
            t=time.perf_counter_ns(); full=attention(mem,q,w,dim,slots); chunked=attention(mem,q,w,dim,slots,chunk); times.append((time.perf_counter_ns()-t)/1e6); maxerr=float(np.max(np.abs(full-chunked)))
        after=snapshot(f'after-{n}',start,'/tmp'); rows.append({'corpus_size':n,'dim':dim,'slots':slots,'dtype':'float32','chunk_size':chunk,'repetitions':reps,'duration_ms':times,'max_abs_chunk_error':maxerr,'before':before.as_dict(),'after':after.as_dict(),'output_shape':list(full.shape)})
    return {'status':'passed','preflight':pre,'results':rows}
if __name__=='__main__':
 print(json.dumps(run(),sort_keys=True))
