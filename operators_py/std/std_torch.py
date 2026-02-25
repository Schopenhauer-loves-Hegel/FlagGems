"""
Torch参考实现 - std
算子类型: reduction
描述: The std operator
"""

import torch

def std(x, dim=None, *, correction=None, keepdim=False):
    return torch.std(x, dim=dim, keepdim=keepdim)