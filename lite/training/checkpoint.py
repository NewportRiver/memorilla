from __future__ import annotations
import json,hashlib
from pathlib import Path

def sha256_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_checkpoint_meta(path,meta):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n'); return sha256_file(p)
