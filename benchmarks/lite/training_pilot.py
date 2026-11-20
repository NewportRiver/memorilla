"""Run only the authorized Android LITE-007 eight-update feasibility pilot."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import json
from lite.training.config import TrainingConfig
from lite.training.train import train_run
DATA=ROOT/'benchmarks/lite/fixtures/lite007'
OUT=Path.home()/'.cache/memorilla-lite/LITE-007-TRAINING-PILOT'
def main():
 cfg=TrainingConfig(batch_size=1,max_updates=8,max_seconds=30.0,max_records_per_example=32,max_threads=2)
 result=train_run(DATA,cfg,updates=8,output_dir=OUT)
 print(json.dumps(result,sort_keys=True,indent=2))
if __name__=='__main__': main()
