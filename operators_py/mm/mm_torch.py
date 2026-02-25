"""
Torch参考实现 - mm
算子类型: blas
描述: The mm operator
"""

import torch

def mm(a, b):
    return torch.mm(a, b)