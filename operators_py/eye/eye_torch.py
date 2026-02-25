"""
Torch参考实现 - eye
算子类型: general
描述: The eye operator
"""

import torch

def eye(size, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    return torch.eye(size, dtype, layout, device, pin_memory)