"""
Torch参考实现 - addr
算子类型: general
描述: The addr operator
"""

import torch

def addr(input, vec1, vec2, *, beta=1, alpha=1):
    return torch.addr(input, vec1, vec2, beta, alpha)