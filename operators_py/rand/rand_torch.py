"""
Torch参考实现 - rand
算子类型: general
描述: The rand operator
"""

import torch

def rand(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.rand(size, dtype, layout, device, pin_memory)