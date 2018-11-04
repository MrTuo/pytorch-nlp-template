import torch
import torch.nn as nn


# 自定义loss函数
def hinge_loss(s1, s2, t0, device):
    zero = torch.zeros((1), device=device)
    loss = torch.sum(torch.max(t0 - s1 + s2, zero))
    return loss
