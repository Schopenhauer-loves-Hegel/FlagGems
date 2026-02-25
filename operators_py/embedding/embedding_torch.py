"""
Torch参考实现 - embedding
算子类型: general
描述: The embedding operator
"""

import torch

def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return torch.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)