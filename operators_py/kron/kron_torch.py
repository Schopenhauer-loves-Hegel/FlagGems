"""
Torch参考实现 - kron
算子类型: general
描述: The kron operator
"""

import torch

def kron(A, B):
    return torch.kron(A, B)