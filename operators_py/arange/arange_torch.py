"""
Torch参考实现 - arange
算子类型: general
描述: The arange operator
"""

import torch

def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.arange(end, dtype, layout, device, pin_memory)