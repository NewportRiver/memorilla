from pathlib import Path
import json
from lite.training.dataset import load_development
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'benchmarks/lite/fixtures/lite007'
def test_scenario_partition_and_counts():
 train,d=load_development(DATA,'train'); val,_=load_development(DATA,'validation'); assert len(train)==960 and len(val)==240; assert {x.scenario_id for x in train}.isdisjoint({x.scenario_id for x in val}); assert len(d['records'])==1500 and len(d['queries'])==1200
def test_embedding_contract_and_no_holdout_labels():
 z=__import__('numpy').load(DATA/'embeddings.npz'); assert z['record_embeddings'].shape==(1500,384); assert z['query_embeddings'].shape==(1200,384); assert not any(DATA.glob('*holdout*'))
