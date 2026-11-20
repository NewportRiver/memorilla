from pathlib import Path
from lite.training.dataset import load_development
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'benchmarks/lite/fixtures/lite007'
def test_policy_and_insufficient_examples_are_distinct():
 ex,_=load_development(DATA); policy=[x for x in ex if x.policy_expected_abstention]; missing=[x for x in ex if x.category=='insufficient_evidence']; assert policy and missing; assert all(not x.positive_mask.any() for x in missing); assert all(x.policy_expected_abstention for x in policy)
def test_answerable_examples_have_required_evidence():
 ex,_=load_development(DATA); answerable=[x for x in ex if x.category in ('multi_hop','multi_evidence') and not x.policy_expected_abstention]; assert answerable and all(x.positive_mask.any() for x in answerable)
