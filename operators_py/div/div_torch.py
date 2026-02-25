"""
Torch参考实现 - div
算子类型: pointwise
描述: The div operator
"""

import torch

def div(input, other):
    return torch.div(input, other, rounding_mode=rounding_mode)