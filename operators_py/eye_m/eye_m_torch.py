"""
Torch参考实现 - eye_m
算子类型: general
描述: The eye_m operator
"""

import torch

def eye_m(n, m, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    return torch.eye_m(n, m, dtype, layout, device, pin_memory)