"""
Torch参考实现 - glu
算子类型: general
描述: The glu operator
"""

import torch

def glu(self, dim=-1):
    return torch.glu(self, dim)