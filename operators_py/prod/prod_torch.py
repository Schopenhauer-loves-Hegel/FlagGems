"""
Torch参考实现 - prod
算子类型: reduction
描述: The prod operator
"""

import torch

def prod(inp, *, dtype=None):
    return torch.prod(inp)