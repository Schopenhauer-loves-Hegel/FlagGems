"""
Torch参考实现 - sum
算子类型: reduction
描述: The sum operator
"""

import torch

def sum(inp, *, dtype=None):
    return torch.sum(inp)