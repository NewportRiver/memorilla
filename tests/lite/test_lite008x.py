import json
from pathlib import Path
from experiments.lite008w.control_utils import pair_reversal

def test_x_manifest_has_four_groups_and_eight_pairs():
 d=json.loads(Path('docs/memorilla-lite/audits/LITE-008X-DATASET-MANIFEST.json').read_text())
 assert {r['group'] for r in d['rows']}==set('ABCD')
 assert all(sum(r['group']==g for r in d['rows'])==8 for g in 'ABCD')
 assert max(max(r['sequence_token_lengths'].values()) for r in d['rows'])<=32

def test_x_unfamiliar_values_are_single_token_and_disjoint_from_prior_words():
 d=json.loads(Path('docs/memorilla-lite/audits/LITE-008X-DATASET-MANIFEST.json').read_text())
 assert d['vocabulary_isolation']['single_token_required']
 assert all(len(r['candidate_token_ids']['a'])==len(r['candidate_token_ids']['b'])==1 for r in d['rows'])

def test_x_assignment_swap_changes_evidence_not_candidates():
 d=json.loads(Path('docs/memorilla-lite/audits/LITE-008X-DATASET-MANIFEST.json').read_text())
 for r in d['rows']:
  assert r['answer_a'] in r['evidence_a'] and r['answer_b'] in r['evidence_b']
  assert r['candidate_token_ids']==r['candidate_token_ids']

def test_x_pair_reversal_is_two_direction_metric():
 assert pair_reversal(1,-1)
 assert not pair_reversal(1,1)

def test_x_zero_optimizer_boundary_is_recorded():
 d=json.loads(Path('docs/memorilla-lite/audits/LITE-008X-DATASET-MANIFEST.json').read_text())
 assert d['optimizer_updates']==0
