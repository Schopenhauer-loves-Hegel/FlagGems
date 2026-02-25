"""
Torch参考实现 - index_add
算子类型: indexing
描述: The index_add operator
"""

import torch

def index_add(inp, dim, index, src, alpha=1):
    return torch.index_add(inp, dim, index, src, alpha)