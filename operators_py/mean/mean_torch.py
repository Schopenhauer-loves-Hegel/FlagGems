"""
Torch参考实现 - mean
算子类型: reduction
描述: The mean operator
"""

import torch

def mean(inp, *, dtype=None):
    return torch.mean(inp)