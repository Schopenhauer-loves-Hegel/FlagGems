"""
Torch参考实现 - contiguous
算子类型: general
描述: The contiguous operator
"""

import torch

def contiguous(inp, memory_format=torch.contiguous_format):
    return torch.contiguous(inp, memory_format)