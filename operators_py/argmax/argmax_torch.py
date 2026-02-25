"""
Torch参考实现 - argmax
算子类型: reduction
描述: The argmax operator
"""

import torch

def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    return torch.argmax(inp, dim=dim, keepdim=keepdim)