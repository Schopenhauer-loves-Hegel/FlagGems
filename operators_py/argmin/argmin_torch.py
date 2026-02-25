"""
Torch参考实现 - argmin
算子类型: reduction
描述: The argmin operator
"""

import torch

def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    return torch.argmin(inp, dim=dim, keepdim=keepdim)