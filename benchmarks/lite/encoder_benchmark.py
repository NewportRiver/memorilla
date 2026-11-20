"""Short bounded benchmark for the single verified MiniLM ONNX artifact."""
from __future__ import annotations
import json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from lite.encoder.onnx_encoder import MiniLMEncoder
from lite.encoder.manifest import model_paths,MODEL_SHA256
from lite.runtime.preflight import ResourceLimits,preflight
from lite.runtime.telemetry import snapshot
TEXT='The compact memory runtime records a verified outcome and preserves evidence provenance.'
def run(reps=2):
 limits=ResourceLimits(min_available_ram_bytes=1024**3,abort_available_ram_bytes=768*1024**2,target_rss_bytes=192*1024**2,max_rss_bytes=256*1024**2,max_storage_bytes=512*1024**2,max_threads=2,max_concurrent_processes=1,timeout_seconds=30); pre=preflight(Path.home(),limits,0)
 if not pre['ok']: return {'status':'blocked','preflight':pre}
 start=time.monotonic_ns(); e=MiniLMEncoder(max_length=128,threads=2); init_ms=(time.monotonic_ns()-start)/1e6; rows=[]
 for length in (16,32,64,128):
  text=' '.join(['memory']*length); durations=[]; before=snapshot(f'before-{length}',start,Path.home()); outs=[]
  for _ in range(reps):
   t=time.perf_counter_ns(); x=e.encode(text); durations.append((time.perf_counter_ns()-t)/1e6); outs.append(x)
  after=snapshot(f'after-{length}',start,Path.home()); rows.append({'requested_words':length,'input_tokens':e.last_metadata['input_tokens'][0],'truncated':e.last_metadata['truncated'][0],'durations_ms':durations,'output_dim':x.shape[0],'norm':float(np.linalg.norm(x)),'repeat_max_abs_error':float(np.max(np.abs(outs[0]-outs[-1]))),'before':before.as_dict(),'after':after.as_dict()})
 e.close(); return {'status':'passed','preflight':pre,'model_sha256':MODEL_SHA256,'model_storage_bytes':model_paths()['model'].stat().st_size,'initialization_ms':init_ms,'results':rows}
if __name__=='__main__': print(json.dumps(run(),sort_keys=True))
