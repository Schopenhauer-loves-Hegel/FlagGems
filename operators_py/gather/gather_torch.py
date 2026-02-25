"""
Torch参考实现 - gather
算子类型: indexing
描述: The gather operator
"""

import torch

def gather(inp, dim, index, out=None, sparse_grad=False):
    return torch.gather(inp, dim, index)