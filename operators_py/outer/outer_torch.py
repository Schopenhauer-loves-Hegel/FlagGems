"""
Torch参考实现 - outer
算子类型: blas
描述: The outer operator
"""

import torch

def outer(inp, weight):
    return torch.outer(inp, weight)