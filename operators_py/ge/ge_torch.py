"""
Torch参考实现 - ge
算子类型: pointwise
描述: The ge operator
"""

import torch

def ge(A, B):
    return torch.ge(A, B)