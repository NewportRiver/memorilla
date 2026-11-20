import torch
from lite.training.objectives import pairwise_loss

def test_multi_positive_loss_has_gradient():
 s=torch.tensor([[0.2,0.1,-0.3]],requires_grad=True); p=torch.tensor([[True,True,False]]); v=torch.tensor([[True,True,True]]); loss=pairwise_loss(s,p,v); loss.backward(); assert loss.item()>0 and s.grad is not None and torch.isfinite(s.grad).all()
def test_empty_positive_is_zero_and_finite():
 s=torch.zeros((1,3),requires_grad=True); p=torch.zeros((1,3),dtype=torch.bool); v=torch.ones((1,3),dtype=torch.bool); loss=pairwise_loss(s,p,v); loss.backward(); assert loss.item()==0.0 and torch.isfinite(s.grad).all()
