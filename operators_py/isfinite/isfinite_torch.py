"""
Torch参考实现 - isfinite
算子类型: pointwise
描述: The isfinite operator
"""

import torch

def isfinite(input):
    return torch.isfinite(input)