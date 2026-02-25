"""
Torch参考实现 - polar
算子类型: general
描述: The polar operator
"""

import torch

def polar(abs, angle):
    return torch.polar(abs, angle)