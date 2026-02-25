"""
Torch参考实现 - celu
算子类型: general
描述: The celu operator
"""

import torch

def celu(A, alpha=1.0):
    return torch.celu(A, alpha)