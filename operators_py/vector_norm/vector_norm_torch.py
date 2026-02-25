"""
Torch参考实现 - vector_norm
算子类型: general
描述: The vector_norm operator
"""

import torch

def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    return torch.vector_norm(x, ord, dim, keepdim, dtype)