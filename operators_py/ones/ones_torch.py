"""
Torch参考实现 - ones
算子类型: general
描述: The ones operator
"""

import torch

def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.ones(size, dtype, layout, device, pin_memory)