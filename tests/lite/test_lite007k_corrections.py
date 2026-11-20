import json
from pathlib import Path
from lite.training.sample_scheduler import PersistentSampleScheduler
from lite.training.evaluation_metrics import ranking_metrics

def test_scheduler_persists_cursor_and_covers_unique_examples(tmp_path):
 p=tmp_path/'scheduler.json'; a=PersistentSampleScheduler(p,'dataset',[0,1,2,3],seed=7)
 first=a.reserve(3,'episode-0'); b=PersistentSampleScheduler(p,'dataset',[0,1,2,3],seed=7)
 second=b.reserve(3,'episode-1')
 assert len(set(first+second))==4
 assert b.reserved_updates==6
 assert json.loads(p.read_text())['cursor']==6

def test_scheduler_rejects_identity_mismatch(tmp_path):
 p=tmp_path/'scheduler.json'; PersistentSampleScheduler(p,'dataset',[0,1],seed=1)
 try: PersistentSampleScheduler(p,'other',[0,1],seed=1)
 except ValueError as e: assert 'identity mismatch' in str(e)
 else: raise AssertionError('identity mismatch accepted')

def test_metrics_distinguish_any_hit_from_complete_evidence():
 r=ranking_metrics([{'ranked_ids':['a','x','b'],'required_ids':['a','b']}])
 assert r['recall_at_1']==1.0
 assert r['recall_at_3']==1.0
 assert r['multi_evidence_coverage']==1.0
 assert r['complete_required_evidence']==1.0
 r=ranking_metrics([{'ranked_ids':['a','x','y'],'required_ids':['a','b']}])
 assert r['recall_at_1']==1.0
 assert r['multi_evidence_coverage']==0.5
 assert r['complete_required_evidence']==0.0
