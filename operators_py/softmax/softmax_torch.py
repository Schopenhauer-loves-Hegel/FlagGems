"""
Torch参考实现 - softmax
算子类型: normalization
描述: The softmax operator
"""

import torch

def softmax(self, dim, half_to_float=False):
    return torch.softmax(self, dim=dim)