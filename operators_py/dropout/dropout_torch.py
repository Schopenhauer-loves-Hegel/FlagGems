"""
Torch参考实现 - dropout
算子类型: general
描述: The dropout operator
"""

import torch

def dropout(input, p, train=True):
    return torch.dropout(input, p, train)