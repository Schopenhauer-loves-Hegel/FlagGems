"""
Torch参考实现 - nonzero
算子类型: general
描述: The nonzero operator
"""

import torch

def nonzero(inp, *, as_tuple=False):
    return torch.nonzero(inp, as_tuple)