from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path

class PersistentSampleScheduler:
    """Crash-safe deterministic cursor over eligible examples.

    Reservations advance before optimizer work so an interrupted episode cannot
    silently replay an example. Separate model schedulers must share the same
    order digest and dataset identity for matched exposure.
    """
    def __init__(self, path, dataset_identity, eligible_indices, seed=0):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        indices=[int(x) for x in eligible_indices]
        if not indices: raise ValueError('scheduler requires eligible examples')
        order=sorted(indices, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
        self.order=order; self.order_digest=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state.get('dataset_identity')!=dataset_identity or self.state.get('order_digest')!=self.order_digest or self.state.get('order')!=order:
                raise ValueError('sample scheduler identity mismatch')
        else:
            self.state={'schema_version':1,'dataset_identity':dataset_identity,'seed':seed,'order':order,'order_digest':self.order_digest,'cursor':0,'reservations':[]}
            self._save()
    def _save(self):
        fd,tmp=tempfile.mkstemp(dir=self.path.parent,suffix='.tmp'); os.close(fd)
        try:
            Path(tmp).write_text(json.dumps(self.state,sort_keys=True,indent=2)+'\n'); os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    def reserve(self,count,episode_id):
        if count<1: raise ValueError('reservation count must be positive')
        start=self.state['cursor']; selected=[]; n=len(self.order)
        for i in range(count): selected.append(self.order[(start+i)%n])
        self.state['cursor']=start+count
        self.state['reservations'].append({'episode_id':episode_id,'start':start,'count':count,'indices':selected})
        self._save(); return selected
    @property
    def reserved_updates(self): return sum(x['count'] for x in self.state['reservations'])
