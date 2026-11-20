import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pytest
from lite.neural import NeuralConfig,NumpyAssociativeMemory,save_model,load_model

def test_roundtrip_and_metadata(tmp_path):
 m=NumpyAssociativeMemory(); p=tmp_path/'weights'; meta=save_model(m,p,{'repo':'encoder','revision':'r1'},'untrained'); x,loaded=load_model(p,{'repo':'encoder','revision':'r1'}); q=np.ones(384,np.float32); mem=np.ones((2,384),np.float32); assert meta['training_status']=='untrained'; assert meta['parameter_count']==429312; assert np.allclose(m.forward(q,mem),x.forward(q,mem))
def test_compatibility_and_corruption_rejected(tmp_path):
 p=tmp_path/'weights'; save_model(NumpyAssociativeMemory(),p,{'revision':'r1'})
 with pytest.raises(ValueError): load_model(p,{'revision':'r2'})
 with np.load(p.with_suffix('.npz'),allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
 arrays['slots']=arrays['slots'].copy(); arrays['slots'].flat[0]+=1
 np.savez(p.with_suffix('.npz'),**arrays)
 with pytest.raises(ValueError): load_model(p)
