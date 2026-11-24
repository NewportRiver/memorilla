import torch
from experiments.lite008k.objective import answer_contrastive_loss

def test_contrastive_objective_promotes_correct_margin_and_float32_loss():
    logits=torch.tensor([[1., 3., 0.]],dtype=torch.bfloat16,requires_grad=True)
    loss,nll,margin=answer_contrastive_loss(logits,1,0,0.5)
    assert loss.dtype==torch.float32 and nll.dtype==torch.float32 and margin.dtype==torch.float32
    assert float(margin.detach())>0 and torch.isfinite(loss)
    loss.backward(); assert logits.grad is not None and torch.isfinite(logits.grad).all()
