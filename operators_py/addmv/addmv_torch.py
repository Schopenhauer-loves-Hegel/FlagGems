"""
Torch参考实现 - addmv
算子类型: general
描述: The addmv operator
"""

import torch

def addmv(self, mat, vec, *, beta=1, alpha=1):
    return torch.addmv(self, mat, vec, beta, alpha)