"""
Torch参考实现 - count_nonzero
算子类型: general
描述: The count_nonzero operator
"""

import torch

def count_nonzero(x, dim=None):
    return torch.count_nonzero(x, dim)