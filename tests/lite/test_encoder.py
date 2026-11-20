import sys,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pytest
from lite.encoder.manifest import model_paths,MODEL_SHA256,OUTPUT_DIM
from lite.encoder.tokenizer import MiniLMTokenizer
from lite.encoder.onnx_encoder import MiniLMEncoder
P=model_paths()
pytestmark=pytest.mark.skipif(not P['model'].exists(),reason='external verified model cache absent')

def test_artifact_checksum_and_contract():
 assert hashlib.sha256(P['model'].read_bytes()).hexdigest()==MODEL_SHA256
 e=MiniLMEncoder(threads=2); x=e.encode('The memory system recovered a failed mission.'); assert x.shape==(OUTPUT_DIM,); assert x.dtype==np.float32; assert np.isfinite(x).all(); assert abs(float(np.linalg.norm(x))-1)<1e-5; e.close()

def test_tokenizer_special_tokens_truncation_and_batch():
 t=MiniLMTokenizer(P['vocab'],max_length=16); x=t.encode('Hello, WORLD!'); assert x['tokens'][:4]==['[CLS]','hello',',','world']; assert x['tokens'][-1]=='[SEP]'; y=t.encode(' '.join(['memory']*100)); assert y['truncated'] and len(y['input_ids'])==16; b=t.batch(['one','two words']); assert np.asarray(b['input_ids']).shape==(2,4)

def test_repeatability_and_batch():
 e=MiniLMEncoder(threads=2); a=e.encode('same semantic text'); b=e.encode('same semantic text'); assert np.max(np.abs(a-b))<1e-6; batch=e.encode_batch(['same semantic text','different text']); assert batch.shape==(2,OUTPUT_DIM); assert np.max(np.abs(batch[0]-batch[1]))>1e-4; e.close()
