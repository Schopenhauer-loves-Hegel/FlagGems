"""
Torch参考实现 - bmm
算子类型: blas
描述: The bmm operator
"""

import torch

def bmm(A, B):
    return torch.bmm(A, B)