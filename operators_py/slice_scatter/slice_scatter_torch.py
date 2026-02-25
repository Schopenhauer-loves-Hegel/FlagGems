"""
Torch参考实现 - slice_scatter
算子类型: general
描述: The slice_scatter operator
"""

import torch

def slice_scatter(inp, src, dim=0, start=None, end=None, step=1):
    return torch.slice_scatter(inp, src, dim, start, end, step)