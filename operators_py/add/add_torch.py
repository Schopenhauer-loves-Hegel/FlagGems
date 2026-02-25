"""
Torch参考实现 - add
算子类型: pointwise
描述: The add operator
"""

import torch

def add(A, B, *, alpha=1):
    return torch.add(A, B, alpha=alpha)