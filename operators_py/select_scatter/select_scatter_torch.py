"""
Torch参考实现 - select_scatter
算子类型: general
描述: The select_scatter operator
"""

import torch

def select_scatter(inp, src, dim, index):
    return torch.select_scatter(inp, src, dim, index)