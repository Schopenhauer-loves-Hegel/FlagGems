"""
Torch参考实现 - cumsum
算子类型: reduction
描述: The cumsum operator
"""

import torch

def cumsum(inp, dim=1, *, dtype=None):
    return torch.cumsum(inp, dim=dim)