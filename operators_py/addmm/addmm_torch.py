"""
Torch参考实现 - addmm
算子类型: blas
描述: The addmm operator
"""

import torch

def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    return torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)