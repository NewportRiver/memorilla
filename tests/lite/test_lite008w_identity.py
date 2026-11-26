import pytest
from experiments.lite008w.identity import canonical_identity,reject_case_target_collision

def base(evidence='Record e stores marble.'):
 return {'manifest_sha256':'m'*64,'partition':'teacher_training','entity':'entity-1','case_id':'case-1','target':'A','evidence':evidence,'query':'Which value?','candidate_answers':['marble','cedar'],'candidate_token_ids':[1,2],'factual_direction':'A','encoder_identity':{'revision':'enc'},'decoder_identity':{'revision':'dec'},'teacher_generation_config':{'updates':16},'artifact_sha256':'a'*64}
def test_complete_identity_is_content_bound():
 assert canonical_identity(base()) != canonical_identity(base('Record e stores cedar.'))
def test_case_target_collision_is_rejected():
 with pytest.raises(ValueError,match='collision'):
  reject_case_target_collision([base(),base('Record e stores cedar.')])
def test_duplicate_full_identity_is_rejected():
 with pytest.raises(ValueError,match='duplicate'):
  from experiments.lite008w.identity import index_records
  index_records([base(),base()])
