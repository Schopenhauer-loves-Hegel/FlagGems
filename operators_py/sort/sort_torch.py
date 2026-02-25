"""
Torch参考实现 - sort
算子类型: general
描述: The sort operator
"""

import torch

def sort(inp, dim=-1, descending=False):
    return torch.sort(inp, dim, descending)