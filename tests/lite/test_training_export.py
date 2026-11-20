import numpy as np
from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.neural.torch_model import TorchAssociativeMemory
from lite.training.export import export_torch,verify_export

def test_export_round_trip(tmp_path):
 t=TorchAssociativeMemory(NeuralConfig()); meta=export_torch(t,tmp_path/'model',{'dimension':384},'test'); m,loaded=verify_export(tmp_path/'model',{'dimension':384}); assert loaded['training_status']=='test'; q=np.ones(384,np.float32); mem=np.ones((3,384),np.float32); assert np.isfinite(m.forward(q,mem)).all(); assert meta['parameter_count']==429312

def test_corrupt_export_rejected(tmp_path):
 t=TorchAssociativeMemory(NeuralConfig()); export_torch(t,tmp_path/'model',{'dimension':384}); z=np.load(tmp_path/'model.npz'); state={k:z[k] for k in z.files}; state['slots']=state['slots'].copy(); state['slots'][0,0]+=1; np.savez(tmp_path/'model.npz',**state)
 try: verify_export(tmp_path/'model',{'dimension':384})
 except ValueError: return
 assert False
