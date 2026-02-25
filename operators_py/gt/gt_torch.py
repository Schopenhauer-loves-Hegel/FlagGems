"""
Torch参考实现 - gt
算子类型: pointwise
描述: The gt operator
"""

import torch

def gt(A, B):
    return torch.gt(A, B)