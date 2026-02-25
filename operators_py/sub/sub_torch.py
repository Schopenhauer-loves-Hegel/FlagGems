"""
Torch参考实现 - sub
算子类型: pointwise
描述: The sub operator
"""

import torch

def sub(A, B, *, alpha=1):
    return torch.sub(A, B, alpha=alpha)