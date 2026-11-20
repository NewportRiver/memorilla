import os, subprocess, sys
import pytest
from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.neural.serialization import save_model, load_model, recover_checkpoint

def test_actual_subprocess_checkpoint_publication_recovery(tmp_path):
    path=tmp_path/'checkpoint'; save_model(NumpyAssociativeMemory(NeuralConfig(seed=1)),path,{'dataset_identity':'d'},'test')
    code="from lite.neural.config import NeuralConfig; from lite.neural.numpy_model import NumpyAssociativeMemory; from lite.neural.serialization import save_model; save_model(NumpyAssociativeMemory(NeuralConfig(seed=2)), r'%s', {'dataset_identity':'d'}, 'fault')" % path
    env=dict(os.environ); env['MEMORILLA_FAULT_CHECKPOINT_PHASE']='after_npz_publish_before_metadata'; env['PYTHONPATH']='.'
    proc=subprocess.run([sys.executable,'-c',code],env=env)
    assert proc.returncode==97
    with pytest.raises((ValueError,KeyError)): load_model(path)
    model,_=recover_checkpoint(path); assert model.config.seed==1
