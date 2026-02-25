"""
Torch参考实现 - bitwise_right_shift
算子类型: general
描述: The bitwise_right_shift operator
"""

import torch

def bitwise_right_shift(self, other, *, out=None):
    return torch.bitwise_right_shift(self, other, out)