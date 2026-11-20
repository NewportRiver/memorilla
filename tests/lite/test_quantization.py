import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pytest
from lite.memory.quantization import encode_vector,decode_vector,cosine

def test_all_formats_roundtrip_and_checksums():
 x=np.linspace(-1,1,384,dtype=np.float32)
 for dtype in ('float32','float16','int8'):
  raw,scale,checksum=encode_vector(x,dtype,384); y=decode_vector(raw,dtype,384,scale,checksum); threshold=0.999 if dtype!='int8' else 0.999; assert cosine(x,y)>threshold

def test_reject_malformed_nonfinite_and_bad_scale():
 with pytest.raises(ValueError): encode_vector(np.full(384,np.nan,np.float32),'float32',384)
 with pytest.raises(ValueError): encode_vector(np.zeros(383,np.float32),'float32',384)
 raw,scale,checksum=encode_vector(np.ones(384,np.float32),'int8',384)
 with pytest.raises(ValueError): decode_vector(raw,'int8',384,0,checksum)
 with pytest.raises(ValueError): decode_vector(raw,'int8',384,scale,checksum[:-1]+'0')
 with pytest.raises(ValueError): decode_vector(raw,'float32',384,scale,checksum)

def test_zero_vector_int8_safe():
 raw,scale,checksum=encode_vector(np.zeros(384,np.float32),'int8',384); assert scale>0; assert np.all(decode_vector(raw,'int8',384,scale,checksum)==0)
