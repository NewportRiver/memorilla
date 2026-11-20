import numpy as np
import pytest
from lite.training.incremental_embeddings import prepare_incremental

class Encoder:
    def __init__(self): self.calls=[]
    def encode_batch(self,texts):
        self.calls.extend(texts); return np.asarray([np.ones(4,np.float32)*len(text) for text in texts])

def dataset(path):
    path.write_text('{"dataset_version":"x","records":[{"evidence_id":"r1","content":"same"},{"evidence_id":"r2","content":"same"}],"queries":[{"query_id":"q1","query":"question"}]}')

def test_incremental_resume_and_duplicate_reuse(tmp_path):
    src=tmp_path/'dataset.json'; dataset(src); out=tmp_path/'embeddings.npz'; progress=tmp_path/'progress.json'; enc=Encoder()
    with pytest.raises(RuntimeError): prepare_incremental(src,out,enc,progress,fail_after=1)
    first=len(enc.calls); meta=prepare_incremental(src,out,enc,progress)
    assert len(enc.calls)==first
    assert meta['reused_texts']==0
    assert out.exists() and out.with_suffix('.json').exists()

def test_incremental_rejects_changed_source(tmp_path):
    src=tmp_path/'dataset.json'; dataset(src); out=tmp_path/'embeddings.npz'; progress=tmp_path/'progress.json'; prepare_incremental(src,out,Encoder(),progress)
    src.write_text(src.read_text().replace('question','changed'))
    with pytest.raises(ValueError): prepare_incremental(src,out,Encoder(),progress)
