__all__ = ["SkewLoss", "skew_loss"]


import torch
import numpy as np
# reg = regularization term
class SkewLoss(torch.nn.Module):
    def forward(self, evals: torch.Tensor) -> torch.Tensor:
        return skew_loss(evals)

# TODO: So far here input should be 1-d always a vector, which means we can only check the skewness on one direction.

def std(evals: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(evals)
    diffs = evals - mean
    var = torch.mean(torch.pow(diffs,2.0))
    return torch.pow(var,0.5)

def skew(evals: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(evals)
    diffs = evals - mean
    var = torch.mean(torch.pow(diffs,2.0))
    std = torch.pow(var,0.5)
    zscores = diffs / std
    return torch.mean(torch.pow(zscores,3.0))

def skew_loss(evals: torch.Tensor) -> torch.Tensor:
    variance = torch.pow(std(evals), 2.0)
    skewness = skew(evals)
    loss = torch.log(1 +  torch.exp(-torch.pow(skewness,2.0)))
    return loss

def test_skew_loss():
    X = torch.rand(100,1)
    print(X)
    skewloss = SkewLoss()
    print(skewloss(X))

if __name__=='__main__':
    test_skew_loss()
