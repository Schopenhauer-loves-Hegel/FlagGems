"""
Torch参考实现 - scatter
算子类型: indexing
描述: The scatter operator
"""

import torch

def scatter(inp, dim, index, src, reduce=None):
    return torch.scatter(inp, dim, index, src)