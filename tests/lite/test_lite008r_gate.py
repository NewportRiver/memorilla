import json
from pathlib import Path
import pytest
from experiments.lite008q.bridge_train import require_qualification

def test_missing_qualification_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match='FAIL_CLOSED'):
        require_qualification(str(tmp_path/'missing.json'))

def test_failed_qualification_fails_closed(tmp_path):
    p=tmp_path/'qualification.json'; p.write_text(json.dumps({'schema':'LITE-008R-gradient-qualification-v1','status':'failed','optimizer_updates':0}))
    with pytest.raises(RuntimeError, match='FAIL_CLOSED'):
        require_qualification(str(p))

def test_incomplete_qualification_fails_closed(tmp_path):
    p=tmp_path/'qualification.json'; p.write_text(json.dumps({'schema':'LITE-008R-gradient-qualification-v1','status':'qualified','optimizer_updates':0,'stage_A':{'status':'pass'}}))
    with pytest.raises(RuntimeError, match='FAIL_CLOSED'):
        require_qualification(str(p))

def test_qualification_cannot_authorize_after_updates(tmp_path):
    p=tmp_path/'qualification.json'; q={'schema':'LITE-008R-gradient-qualification-v1','status':'qualified','optimizer_updates':1,**{s:{'status':'pass'} for s in ('stage_A','stage_B','stage_C','stage_D')}}; p.write_text(json.dumps(q))
    with pytest.raises(RuntimeError, match='FAIL_CLOSED'):
        require_qualification(str(p))
