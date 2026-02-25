"""
Torch参考实现 - logical_xor
算子类型: general
描述: The logical_xor operator
"""

import torch

def logical_xor(A, B):
    return torch.logical_xor(A, B)