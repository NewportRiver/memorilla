from __future__ import annotations
from dataclasses import dataclass,asdict
@dataclass(frozen=True)
class TrainingConfig:
 dataset_version:str='lite007-dev-v1'
 seed:int=20261120
 learning_rate:float=3e-4
 weight_decay:float=0.01
 grad_clip:float=1.0
 batch_size:int=4
 max_epochs:int=10
 max_updates:int=3000
 max_seconds:float=1800.0
 patience:int=3
 max_records_per_example:int=32
 max_threads:int=2
 min_available_ram_bytes:int=1024**3
 abort_available_ram_bytes:int=768*1024**2
 max_rss_bytes:int=256*1024**2
 def as_dict(self): return asdict(self)
