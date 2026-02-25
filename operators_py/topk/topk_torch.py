"""
Torch参考实现 - topk
算子类型: general
描述: The topk operator
"""

import torch

def topk(x, k, dim=-1, largest=True, sorted=True):
    return torch.topk(x, k, dim, largest, sorted)