"""
Torch参考实现 - randn
算子类型: general
描述: The randn operator
"""

import torch

def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.randn(size, dtype, layout, device, pin_memory)