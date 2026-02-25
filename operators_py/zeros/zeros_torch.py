"""
Torch参考实现 - zeros
算子类型: general
描述: The zeros operator
"""

import torch

def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.zeros(size, dtype, layout, device, pin_memory)