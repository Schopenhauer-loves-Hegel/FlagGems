"""
Torch参考实现 - baddbmm
算子类型: blas
描述: The baddbmm operator
"""

import torch

def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return torch.baddbmm(bias, A, B, beta=beta, alpha=alpha)