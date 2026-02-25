"""
Torch参考实现 - bitwise_left_shift
算子类型: general
描述: The bitwise_left_shift operator
"""

import torch

def bitwise_left_shift(self, other, *, out=None):
    return torch.bitwise_left_shift(self, other, out)