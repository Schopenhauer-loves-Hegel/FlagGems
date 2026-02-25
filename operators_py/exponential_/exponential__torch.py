"""
Torch参考实现 - exponential_
算子类型: general
描述: The exponential_ operator
"""

import torch

def exponential_(x, lambd: float = 1.0, *, generator=None):
    return torch.exponential_(x, lambd: float, generator)