"""
Torch参考实现 - pad
算子类型: general
描述: The pad operator
"""

import torch

def pad(self, pad, mode="constant", value=None):
    return torch.pad(self, pad, mode, value)