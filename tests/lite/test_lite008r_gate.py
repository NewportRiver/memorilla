import json
from pathlib import Path
import pytest
from experiments.lite008q.bridge_train import require_qualification,train_bridge

def valid():
 return {'schema':'LITE-008S-gradient-qualification-v1','status':'qualified','optimizer_updates':0,'stages':{s:{'status':'pass'} for s in ('stage_A','stage_B','stage_C','stage_D')},'decoder_artifact_identity':{'id':'d'},'tokenizer_identity':{'id':'t'},'encoder_identity':{'id':'e'},'bridge':'Linear(384,576)','qualification_implementation_sha256':'x','objective':'contrastive','numerical_configuration':{'threads':1},'input_identity':{'id':'i'}}

def test_missing_qualification_fails_closed(tmp_path):
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(tmp_path/'missing.json'))

def test_failed_qualification_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();q['status']='failed';p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p))

def test_incomplete_qualification_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();del q['decoder_artifact_identity'];p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p))

def test_architecture_or_objective_mismatch_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();q['objective']='answer-token cross-entropy';p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p),'contrastive')

def test_post_update_qualification_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();q['optimizer_updates']=1;p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p))

def test_optimizer_boundary_checks_before_model_use(tmp_path):
 p=tmp_path/'q.json';q=valid();q['status']='failed';p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): train_bridge(None,None,None,None,'contrastive',1,.1,None,str(p))
