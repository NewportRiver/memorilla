import json
import pytest
from experiments.lite008q.bridge_train import require_qualification,train_bridge

def valid():
 return {'schema':'LITE-008T-contrastive-gradient-qualification-v1','status':'qualified','optimizer_updates':0,'cases':[{'external_embedding':{'reference_relative_l2':0.0,'cosine':1.0}}],'bridge_parameter_rows':[{'metrics':{'reference_relative_error':0.0}}],'decoder_hashes':{'model.safetensors':'x'},'tokenizer_hashes':{'tokenizer.json':'x'},'encoder_identity':{'sha256':'x'},'bridge_architecture':'Linear(384,576)','parameterization':'residual_eos_plus_linear','objective_implementation_hash':'x','decoder_differentiation_hash':'x','objective':'answer_NLL + 0.5*softplus(-correct_margin)','numerical_configuration':{'threads':1},'input_identity':{'id':'i'},'acceptance_contract_identity':'LITE-008T-contract-v1'}

def test_missing_qualification_fails_closed(tmp_path):
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(tmp_path/'missing.json'))
def test_failed_qualification_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();q['status']='failed';p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p))
def test_incomplete_identity_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();del q['decoder_hashes'];p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p))
def test_changed_objective_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();q['objective']='nll';p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p),'contrastive')
def test_changed_parameterization_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();q['parameterization']='plain_linear';p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p))
def test_post_update_fails_closed(tmp_path):
 p=tmp_path/'q.json';q=valid();q['optimizer_updates']=1;p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): require_qualification(str(p))
def test_optimizer_boundary_checks_before_model_use(tmp_path):
 p=tmp_path/'q.json';q=valid();q['status']='failed';p.write_text(json.dumps(q))
 with pytest.raises(RuntimeError,match='FAIL_CLOSED'): train_bridge(None,None,None,None,'contrastive',1,.1,None,str(p))
