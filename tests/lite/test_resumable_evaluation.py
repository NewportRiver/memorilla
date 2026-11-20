import pytest
from lite.training.resumable_evaluation import ResumableEvaluation

def test_resume_and_complete_only_after_all_batches(tmp_path):
    ids=[f'q{i}' for i in range(5)]
    ev=ResumableEvaluation(tmp_path,'e','m','d','validation',ids,batch_size=2)
    with pytest.raises(RuntimeError): ev.run(lambda xs:[{'ok':x} for x in xs],fail_after_batches=1)
    ev=ResumableEvaluation(tmp_path,'e','m','d','validation',ids,batch_size=2)
    with pytest.raises(RuntimeError): ev.aggregate()
    result=ev.run(lambda xs:[{'ok':x} for x in xs]); assert result['denominator']==5

def test_identity_and_duplicate_protection(tmp_path):
    ids=['q0']; ev=ResumableEvaluation(tmp_path,'e','m','d','validation',ids)
    ev.run(lambda xs:[1 for _ in xs])
    with pytest.raises(ValueError): ResumableEvaluation(tmp_path,'e','different','d','validation',ids)
