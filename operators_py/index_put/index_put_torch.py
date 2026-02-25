"""
Torch参考实现 - index_put
算子类型: general
描述: The index_put operator
"""

import torch

def index_put(inp, indices, values, accumulate=False):
    return torch.index_put(inp, indices, values, accumulate)