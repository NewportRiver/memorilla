import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pytest
from lite.encoder.manifest import model_paths
from lite.encoder.onnx_encoder import MiniLMEncoder
from lite.attention.reference import attention,deterministic_weights
pytestmark=pytest.mark.skipif(not model_paths()['model'].exists(),reason='external verified model cache absent')

def test_encoder_to_attention_contract():
 e=MiniLMEncoder(threads=2); mem=e.encode_batch(['a recovered mission outcome','a failed runtime experiment','unrelated weather report']); q=e.encode('runtime failure'); w=deterministic_weights(384,8); full=attention(mem,q,w,384,8); chunk=attention(mem,q,w,384,8,2); assert mem.shape==(3,384); assert full.shape==(8,384); assert np.isfinite(full).all(); assert np.max(np.abs(full-chunk))<1e-5; e.close()
