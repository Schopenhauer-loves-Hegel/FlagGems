"""
Torch参考实现 - amax
算子类型: reduction
描述: The amax operator
"""

import torch

def amax(inp, dim=None, keepdim=False):
    return torch.amax(inp, dim=dim, keepdim=keepdim)