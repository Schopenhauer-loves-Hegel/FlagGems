"""
Torch参考实现 - nan_to_num
算子类型: general
描述: The nan_to_num operator
"""

import torch

def nan_to_num(A, nan=None, posinf=None, neginf=None):
    return torch.nan_to_num(A, nan, posinf, neginf)