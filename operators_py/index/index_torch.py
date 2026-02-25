"""
Torch参考实现 - index
算子类型: general
描述: The index operator
"""

import torch

def index(inp, indices):
    return torch.index(inp, indices)