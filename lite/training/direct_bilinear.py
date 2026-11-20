from __future__ import annotations
import hashlib,json,os,tempfile,time,resource
from pathlib import Path
import numpy as np
from lite.training.prepared_dataset import PreparedTrainingSet
from lite.training.orchestration import EpisodeBudget
from lite.training.sample_scheduler import PersistentSampleScheduler
from lite.training.numpy_telemetry import available_ram_bytes, rss_bytes

def _sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(path,W,meta):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); npz=path.with_suffix('.npz'); js=path.with_suffix('.json'); fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.npz'); os.close(fd)
 try:
  with open(tmp,'wb') as f: np.savez(f,W=W); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,npz); m={'schema_version':1,'model':'direct-bilinear-v1','parameter_count':int(W.size),'dtype':'float32','shape':list(W.shape),'sha256':_sha(W),**meta}; mt=js.with_suffix('.json.tmp'); mt.write_text(json.dumps(m,sort_keys=True,indent=2)+'\n'); os.replace(mt,js); return m
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def load(path):
 path=Path(path); m=json.loads(path.with_suffix('.json').read_text());
 with np.load(path.with_suffix('.npz'),allow_pickle=False) as z: W=np.asarray(z['W'],np.float32)
 if list(W.shape)!=m['shape'] or _sha(W)!=m['sha256'] or not np.isfinite(W).all(): raise ValueError('direct checkpoint integrity failure')
 return W,m

def create_initial_checkpoint(path, dimension=384, mode='zeros', seed=None):
 if mode!='zeros': raise ValueError('only explicit zero initialization is currently supported')
 W=np.zeros((dimension,dimension),np.float32)
 return save(path,W,{'initialization_mode':mode,'initialization_seed':seed,'provenance':'LITE-007K-corrected-preparation'})

def inference_telemetry(prepared_path, checkpoint_path, examples=8):
 ds=PreparedTrainingSet(prepared_path, allow_validation=True); W,_=load(checkpoint_path); start=time.perf_counter(); rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; count=0
 try:
  for index in ds.eligible_indices()[:examples]:
   ex=ds.get(index); q=ex['query_embedding']; mem=ex['memory_embeddings']; _=q@W@mem.T; count+=1
 finally: ds.close()
 rss1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return {'operation':'direct_bilinear_inference_probe','examples':count,'wall_seconds':time.perf_counter()-start,'ru_maxrss_kib':rss1,'ru_maxrss_delta_kib':max(0,rss1-rss0),'threads':1,'computational_subprocesses':1}
def run_episode(prepared_path,out,budget_path,identity,init=None,requested=8,total=240,scheduler_path=None,scheduler_seed=0,schedule_order=None):
 if str(identity).startswith(('LITE-007L','LITE-008')) and scheduler_path is None: raise ValueError('persistent scheduler is mandatory for corrected comparative experiments')
 out=Path(out); ds=PreparedTrainingSet(prepared_path); started=time.perf_counter(); rss_start=rss_bytes(); ident=ds.metadata['content_sha256'];
 if available_ram_bytes() and available_ram_bytes() < 1280*1024**2: ds.close(); raise RuntimeError('minimum RAM guard blocked direct episode')
 if rss_bytes() > 256*1024**2: ds.close(); raise RuntimeError('direct RSS guard exceeded')
 b=EpisodeBudget(budget_path,total,ident,identity); eid=out.name; b.begin_episode(eid,requested)
 if init is None: raise ValueError('explicit direct initialization checkpoint required')
 W,init_meta=load(init); scheduler=PersistentSampleScheduler(scheduler_path,ident,ds.eligible_indices(),scheduler_seed,schedule_order) if scheduler_path is not None else None; usable=scheduler.reserve(requested,eid) if scheduler is not None else ds.eligible_indices()[:requested]; rows=[]
 try:
  for pos in range(requested):
   if time.perf_counter()-started > 30: raise TimeoutError('direct episode wall-time guard exceeded')
   if available_ram_bytes() and available_ram_bytes() < 1024*1024**2: raise MemoryError('direct abort-RAM guard exceeded')
   if rss_bytes() > 256*1024**2: raise MemoryError('direct RSS guard exceeded')
   ex=ds.get(usable[pos])
   q=ex['query_embedding']; mem=ex['memory_embeddings']; valid=ex['valid_mask']; positive=ex['positive_mask']; pi=int(np.flatnonzero(positive)[0]); ni=int(np.flatnonzero(valid & ~positive)[0]); ps=float(q@W@mem[pi]); ns=float(q@W@mem[ni]); z=np.float32(0.1+ns-ps); loss=float(np.logaddexp(0,z)); coeff=np.float32(1/(1+np.exp(-float(z)))); grad=coeff*(np.outer(q,mem[ni])-np.outer(q,mem[pi])); norm=float(np.linalg.norm(grad)); scale=min(1.0,1.0/max(norm,1e-12)); grad=(grad*scale).astype(np.float32); b.record_update(eid); W=(W-np.float32(.001)*grad).astype(np.float32); save(out/'checkpoint',W,{'dataset_identity':ident,'experiment_identity':identity,'training_status':'direct_bilinear_sgd'}); rows.append({'position':pos,'query_id':ex['query_id'],'loss':loss,'gradient_norm':norm,'clip_scale':scale,'checkpoint_sha256':_sha(W)})
 except BaseException:
  b.finish_episode(eid,'interrupted'); ds.close(); raise
 b.finish_episode(eid,'completed'); ds.close(); result={'episode':eid,'updates':len(rows),'rows':rows,'checkpoint':str(out/'checkpoint'),'final_checksum':_sha(W),'parameter_count':int(W.size),'dataset_identity':ident,'initialization':init_meta,'telemetry':{'absolute_peak_rss_bytes':rss_bytes(),'rss_delta_bytes':max(0,rss_bytes()-rss_start),'available_ram_bytes':available_ram_bytes(),'wall_seconds':time.perf_counter()-started,'numerical_threads':int(os.environ.get('OMP_NUM_THREADS','1')),'computational_subprocesses':1,'resource_guard_status':'passed'}}; (out/'run.json.tmp').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n'); os.replace(out/'run.json.tmp',out/'run.json'); return result
