"""Precompute pinned MiniLM embeddings for the LITE-007 development corpus."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from lite.encoder.onnx_encoder import MiniLMEncoder
from lite.encoder.manifest import MODEL_REPO,MODEL_REVISION,MODEL_SHA256,OUTPUT_DIM
from lite.memory.provenance import EncoderContract
DATA=ROOT/'benchmarks/lite/fixtures/lite007'; OUT=DATA/'embeddings.npz'

def run(batch_size=32):
 d=json.loads((DATA/'dataset.json').read_text()); enc=MiniLMEncoder(threads=2); texts=[x['content'] for x in d['records']]+[x['query'] for x in d['queries']]; out=[]
 for i in range(0,len(texts),batch_size): out.append(enc.encode_batch(texts[i:i+batch_size]))
 arr=np.concatenate(out,axis=0).astype('float32'); n=len(d['records']); assert arr.shape==(n+len(d['queries']),OUTPUT_DIM); assert np.isfinite(arr).all()
 np.savez_compressed(OUT,record_ids=np.asarray([x['evidence_id'] for x in d['records']],dtype='U64'),query_ids=np.asarray([x['query_id'] for x in d['queries']],dtype='U64'),record_embeddings=arr[:n],query_embeddings=arr[n:],encoder_json=json.dumps({'repo':MODEL_REPO,'revision':MODEL_REVISION,'sha256':MODEL_SHA256,'dimension':OUTPUT_DIM,'contract':'attention-mask-mean-l2-v1','dtype':'float32'}))
 meta={'dataset_version':d['dataset_version'],'encoder':{'repo':MODEL_REPO,'revision':MODEL_REVISION,'sha256':MODEL_SHA256,'dimension':OUTPUT_DIM,'contract':'attention-mask-mean-l2-v1'},'record_count':n,'query_count':len(d['queries']),'dtype':'float32','path':str(OUT)}
 (DATA/'embeddings.manifest.json').write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n'); enc.close(); print(json.dumps(meta,sort_keys=True,indent=2))
if __name__=='__main__': run()
