"""
Torch参考实现 - full
算子类型: general
描述: The full operator
"""

import torch

def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.full(size, fill_value, dtype, layout, device, pin_memory)