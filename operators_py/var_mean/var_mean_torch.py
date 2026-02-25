"""
Torch参考实现 - var_mean
算子类型: general
描述: The var_mean operator
"""

import torch

def var_mean(x, dim=None, *, correction=None, keepdim=False):
    return torch.var_mean(x, dim, correction, keepdim)