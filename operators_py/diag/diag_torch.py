"""
Torch参考实现 - diag
算子类型: general
描述: The diag operator
"""

import torch

def diag(x, diagonal=0):
    return torch.diag(x, diagonal)