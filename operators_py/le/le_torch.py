"""
Torch参考实现 - le
算子类型: pointwise
描述: The le operator
"""

import torch

def le(A, B):
    return torch.le(A, B)