"""
Torch参考实现 - triu
算子类型: general
描述: The triu operator
"""

import torch

def triu(A, diagonal=0):
    return torch.triu(A, diagonal)