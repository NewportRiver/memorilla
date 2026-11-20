from __future__ import annotations
from pathlib import Path
import numpy as np
from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.neural.serialization import save_model,load_model

def export_torch(model,path,encoder_compatibility,training_status='trained_pilot'):
 state=model.export_numpy_state(); cfg=NeuralConfig(**model.config.as_dict()); out=NumpyAssociativeMemory(cfg,state=state); return save_model(out,Path(path),encoder_compatibility,training_status)
def verify_export(path,encoder_compatibility): return load_model(path,expected_encoder_compatibility=encoder_compatibility)
