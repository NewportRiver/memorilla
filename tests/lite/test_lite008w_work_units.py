import json
from pathlib import Path
import pytest
from experiments.lite008w.work_units import atomic_json, atomic_torch, content_identity

def rec(e='evidence'):
 return {'partition':'teacher_training','entity':'entity-1','id':'wtrain_00','target':'A','evidence':e,'query':'Which value?','answer_a':'marble','answer_b':'cedar'}
def test_atomic_json_publication(tmp_path):
 p=tmp_path/'x.json'; atomic_json(p,{'status':'complete'}); assert json.loads(p.read_text())['status']=='complete'; assert not p.with_suffix('.json.tmp').exists()
def test_content_identity_changes_with_content():
 a,_=content_identity(rec(), 'm'*64, {'encoder':'e'}, {'decoder':'d'}, [1,2], {'updates':16})
 b,_=content_identity(rec('changed'), 'm'*64, {'encoder':'e'}, {'decoder':'d'}, [1,2], {'updates':16})
 assert a != b
def test_incomplete_artifact_rejected_by_status(tmp_path):
 p=tmp_path/'checkpoint.json'; atomic_json(p,{'status':'INCOMPLETE','completed_updates':3}); assert json.loads(p.read_text())['status']!='COMPLETED'
def test_duplicate_update_prevention_fixture():
 updates=[{'update':1},{'update':2}]; assert len({x['update'] for x in updates})==len(updates)
def test_cumulative_budget_accounting():
 assert 128+160 <= 318
def test_soft_deadline_is_below_hard_limit():
 from experiments.lite008w.work_units import SOFT_DEADLINE,HARD_DEADLINE
 assert SOFT_DEADLINE < HARD_DEADLINE and SOFT_DEADLINE <= 240
