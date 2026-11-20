from __future__ import annotations
import hashlib,json,os,tempfile,time,resource
from pathlib import Path
import numpy as np
from lite.training.prepared_dataset import PreparedTrainingSet
from lite.training.orchestration import EpisodeBudget

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

def inference_telemetry(prepared_path, checkpoint_path, examples=8):
 ds=PreparedTrainingSet(prepared_path, allow_validation=True); W,_=load(checkpoint_path); start=time.perf_counter(); rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; count=0
 try:
  for index in ds.eligible_indices()[:examples]:
   ex=ds.get(index); q=ex['query_embedding']; mem=ex['memory_embeddings']; _=q@W@mem.T; count+=1
 finally: ds.close()
 rss1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return {'operation':'direct_bilinear_inference_probe','examples':count,'wall_seconds':time.perf_counter()-start,'ru_maxrss_kib':rss1,'ru_maxrss_delta_kib':max(0,rss1-rss0),'threads':1,'computational_subprocesses':1}
def run_episode(prepared_path,out,budget_path,identity,init=None,requested=8,total=240):
 out=Path(out); ds=PreparedTrainingSet(prepared_path); ident=ds.metadata['content_sha256']; b=EpisodeBudget(budget_path,total,ident,identity); eid=out.name; b.begin_episode(eid,requested); W=load(init)[0] if init else np.zeros((384,384),np.float32); rows=[]
 try:
  usable=ds.eligible_indices()
  for pos in range(requested):
   ex=ds.get(usable[pos])
   q=ex['query_embedding']; mem=ex['memory_embeddings']; valid=ex['valid_mask']; positive=ex['positive_mask']; pi=int(np.flatnonzero(positive)[0]); ni=int(np.flatnonzero(valid & ~positive)[0]); ps=float(q@W@mem[pi]); ns=float(q@W@mem[ni]); z=np.float32(0.1+ns-ps); loss=float(np.logaddexp(0,z)); coeff=np.float32(1/(1+np.exp(-float(z)))); grad=coeff*(np.outer(q,mem[ni])-np.outer(q,mem[pi])); norm=float(np.linalg.norm(grad)); scale=min(1.0,1.0/max(norm,1e-12)); grad=(grad*scale).astype(np.float32); b.record_update(eid); W=(W-np.float32(.001)*grad).astype(np.float32); save(out/'checkpoint',W,{'dataset_identity':ident,'experiment_identity':identity,'training_status':'direct_bilinear_sgd'}); rows.append({'position':pos,'query_id':ex['query_id'],'loss':loss,'gradient_norm':norm,'clip_scale':scale,'checkpoint_sha256':_sha(W)})
 except BaseException:
  b.finish_episode(eid,'interrupted'); ds.close(); raise
 b.finish_episode(eid,'completed'); ds.close(); result={'episode':eid,'updates':len(rows),'rows':rows,'checkpoint':str(out/'checkpoint'),'final_checksum':_sha(W),'parameter_count':int(W.size),'dataset_identity':ident}; (out/'run.json.tmp').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n'); os.replace(out/'run.json.tmp',out/'run.json'); return result
