from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from .config import NeuralConfig
from .numpy_model import NumpyAssociativeMemory

def _sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save_model(model,path,encoder_compatibility,training_status='untrained'):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); npz=path.with_suffix('.npz'); meta_path=path.with_suffix('.json'); arrays=model.state
 checks={k:_sha(v) for k,v in sorted(arrays.items())}; meta={'schema_version':1,'architecture_version':model.config.architecture_version,'config':model.config.as_dict(),'parameter_count':model.parameter_count(),'weight_bytes':sum(v.nbytes for v in arrays.values()),'dtypes':{k:str(v.dtype) for k,v in arrays.items()},'shapes':{k:list(v.shape) for k,v in arrays.items()},'checksums':checks,'encoder_compatibility':encoder_compatibility,'initialization_seed':model.config.seed,'training_status':training_status,'executable_serialization':False}
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.npz',dir=str(path.parent)); os.close(fd)
 try:
  np.savez(tmp,**arrays); os.replace(tmp,npz)
  meta_tmp=meta_path.with_suffix(meta_path.suffix+'.tmp')
  meta_tmp.write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n'); os.replace(meta_tmp,meta_path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
 return meta

def load_model(path,expected_encoder_compatibility=None):
 path=Path(path); meta=json.loads(path.with_suffix('.json').read_text());
 if meta.get('schema_version')!=1 or meta.get('executable_serialization') is not False: raise ValueError('unsupported neural weight schema')
 if expected_encoder_compatibility is not None and meta.get('encoder_compatibility')!=expected_encoder_compatibility: raise ValueError('encoder compatibility mismatch')
 with np.load(path.with_suffix('.npz'),allow_pickle=False) as z: state={k:np.asarray(z[k],np.float32) for k in z.files}
 if set(state)!=set(meta['shapes']): raise ValueError('parameter set mismatch')
 for k,x in state.items():
  if list(x.shape)!=meta['shapes'][k] or str(x.dtype)!=meta['dtypes'][k] or _sha(x)!=meta['checksums'][k] or not np.isfinite(x).all(): raise ValueError(f'weight integrity failure: {k}')
 model=NumpyAssociativeMemory(NeuralConfig(**meta['config']),state=state)
 if model.parameter_count()!=meta['parameter_count']: raise ValueError('parameter count mismatch')
 return model,meta
