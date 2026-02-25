"""
Torch参考实现 - index_select
算子类型: indexing
描述: The index_select operator
"""

import torch

def index_select(inp, dim, index):
    return torch.index_select(inp, dim, index)